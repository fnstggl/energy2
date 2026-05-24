"""Optimization action vocabulary for SLA evaluation.

These are the abstract optimization moves the correction engine reasons about,
independent of how a particular optimizer (greedy scheduler, MILP, runtime
controller) represents them internally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ActionType(str, Enum):
    MIGRATE = "migrate_workload"
    REROUTE = "reroute_workload"
    DEFER = "defer_workload"
    SCALE_REPLICAS = "scale_replicas"
    CONSOLIDATE = "consolidate_workloads"
    SPREAD = "spread_workloads"
    CHOOSE_CHEAPER_REGION = "choose_cheaper_region"
    CHOOSE_LOWER_CARBON_REGION = "choose_lower_carbon_region"
    CHANGE_PLACEMENT = "change_placement"
    KEEP = "keep_current_placement"


# Actions that physically move a workload between regions/nodes and therefore
# count as a "migration" for migration-governance constraints.
MIGRATION_ACTIONS = frozenset(
    {
        ActionType.MIGRATE,
        ActionType.REROUTE,
        ActionType.CHOOSE_CHEAPER_REGION,
        ActionType.CHOOSE_LOWER_CARBON_REGION,
        ActionType.CHANGE_PLACEMENT,
    }
)

# Actions that change replica packing and therefore affect queue/utilization.
PACKING_ACTIONS = frozenset({ActionType.CONSOLIDATE, ActionType.SPREAD, ActionType.SCALE_REPLICAS})


@dataclass
class OptimizationAction:
    """A candidate optimization action under consideration.

    Attributes:
        action_type: The kind of move.
        target_region: Region the action would place the workload in (if any).
        expected_savings_pct: Unconstrained expected cost savings vs current
            placement, in percent. Positive = cheaper. This is what the
            optimizer is trying to maximize before SLA correction.
        target_replicas: Desired replica count (for scale/consolidate/spread).
        description: Optional human description.
    """

    action_type: ActionType
    target_region: Optional[str] = None
    expected_savings_pct: float = 0.0
    target_replicas: Optional[int] = None
    description: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def is_migration(self) -> bool:
        # Only counts as a migration if it actually changes region.
        if self.action_type not in MIGRATION_ACTIONS:
            return False
        return True

    @property
    def is_noop(self) -> bool:
        return self.action_type == ActionType.KEEP


def keep_current(region: Optional[str] = None) -> OptimizationAction:
    """The canonical no-op action: keep the current placement, zero savings."""
    return OptimizationAction(
        action_type=ActionType.KEEP,
        target_region=region,
        expected_savings_pct=0.0,
        description="Keep current placement (no-op)",
    )
