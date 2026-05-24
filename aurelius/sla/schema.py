"""SLA policy schema for Aurelius.

Defines the data model for hard and soft Service Level Agreements (SLAs) that
constrain optimization decisions, plus the priority-tier system that supplies
sensible defaults.

Design notes
------------
* Plain dataclasses (no pydantic) so the schema is dependency-free and the
  tests run anywhere. Validation is explicit and returns useful errors via
  :class:`SLAValidationError`.
* Every constraint field is ``Optional`` and defaults to ``None`` meaning
  "unspecified". Unspecified fields are filled from the workload's priority
  tier defaults at load time (see :func:`apply_tier_defaults`). A field that
  is still ``None`` after tier merge is simply not enforced — Aurelius never
  invents a guarantee it was not given.
* Hard constraints BLOCK an action when violated. Soft constraints only
  reduce an action's score.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Optional


class SLAValidationError(ValueError):
    """Raised when an SLA policy config is malformed.

    Carries the full list of human-readable problems so a caller can surface
    all of them at once instead of one-at-a-time.
    """

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("Invalid SLA policy:\n  - " + "\n  - ".join(errors))


class PriorityTier(str, Enum):
    """Workload priority tiers, ordered safest -> most cost-optimized."""

    CRITICAL = "critical"
    LATENCY_SENSITIVE = "latency_sensitive"
    STANDARD = "standard"
    FLEXIBLE = "flexible"
    BATCH = "batch"


class OptimizationAggressiveness(str, Enum):
    """How aggressively the optimizer may trade SLA headroom for savings."""

    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"


# ---------------------------------------------------------------------------
# Hard constraints — violating any of these BLOCKS the action.
# ---------------------------------------------------------------------------
@dataclass
class HardSLA:
    """Hard SLA constraints. A violation makes an action disallowed.

    All fields optional; ``None`` means "not enforced".
    """

    # Placement / residency
    allowed_regions: Optional[list[str]] = None
    forbidden_regions: Optional[list[str]] = None
    data_residency_region: Optional[str] = None

    # Latency / queueing
    max_p95_latency_ms: Optional[float] = None
    max_p99_latency_ms: Optional[float] = None
    max_queue_wait_ms: Optional[float] = None

    # Reliability
    min_availability_pct: Optional[float] = None
    max_error_rate_pct: Optional[float] = None
    max_timeout_rate_pct: Optional[float] = None

    # Migration governance
    migration_allowed: Optional[bool] = None
    max_migrations_per_hour: Optional[int] = None
    # List of [start_iso, end_iso] windows during which migration is forbidden.
    no_migration_windows: Optional[list[list[str]]] = None

    # Capacity
    required_capacity_buffer_pct: Optional[float] = None

    def validate(self, prefix: str = "hard") -> list[str]:
        errs: list[str] = []

        def _pct(name: str, val: Optional[float]):
            if val is not None and not (0.0 <= float(val) <= 100.0):
                errs.append(f"{prefix}.{name} must be between 0 and 100, got {val}")

        def _nonneg(name: str, val):
            if val is not None and float(val) < 0:
                errs.append(f"{prefix}.{name} must be >= 0, got {val}")

        _nonneg("max_p95_latency_ms", self.max_p95_latency_ms)
        _nonneg("max_p99_latency_ms", self.max_p99_latency_ms)
        _nonneg("max_queue_wait_ms", self.max_queue_wait_ms)
        _pct("min_availability_pct", self.min_availability_pct)
        _pct("max_error_rate_pct", self.max_error_rate_pct)
        _pct("max_timeout_rate_pct", self.max_timeout_rate_pct)
        _pct("required_capacity_buffer_pct", self.required_capacity_buffer_pct)

        if (
            self.max_p95_latency_ms is not None
            and self.max_p99_latency_ms is not None
            and self.max_p99_latency_ms < self.max_p95_latency_ms
        ):
            errs.append(
                f"{prefix}.max_p99_latency_ms ({self.max_p99_latency_ms}) must be "
                f">= max_p95_latency_ms ({self.max_p95_latency_ms})"
            )

        if self.max_migrations_per_hour is not None and self.max_migrations_per_hour < 0:
            errs.append(
                f"{prefix}.max_migrations_per_hour must be >= 0, "
                f"got {self.max_migrations_per_hour}"
            )

        for rname in ("allowed_regions", "forbidden_regions"):
            val = getattr(self, rname)
            if val is not None and not isinstance(val, list):
                errs.append(f"{prefix}.{rname} must be a list of region strings")

        # allowed vs forbidden contradiction
        if self.allowed_regions and self.forbidden_regions:
            overlap = set(self.allowed_regions) & set(self.forbidden_regions)
            if overlap:
                errs.append(
                    f"{prefix}: regions {sorted(overlap)} are in both "
                    f"allowed_regions and forbidden_regions"
                )

        if (
            self.data_residency_region is not None
            and self.allowed_regions
            and self.data_residency_region not in self.allowed_regions
        ):
            errs.append(
                f"{prefix}.data_residency_region '{self.data_residency_region}' "
                f"is not in allowed_regions {self.allowed_regions}"
            )

        if self.no_migration_windows is not None:
            if not isinstance(self.no_migration_windows, list):
                errs.append(f"{prefix}.no_migration_windows must be a list of [start, end] pairs")
            else:
                for i, w in enumerate(self.no_migration_windows):
                    if not isinstance(w, (list, tuple)) or len(w) != 2:
                        errs.append(
                            f"{prefix}.no_migration_windows[{i}] must be a "
                            f"[start_iso, end_iso] pair"
                        )
        return errs


# ---------------------------------------------------------------------------
# Soft constraints — violating these only penalizes the action's score.
# ---------------------------------------------------------------------------
@dataclass
class SoftSLA:
    """Soft SLA preferences. Violations reduce an action's score but do not block."""

    preferred_regions: Optional[list[str]] = None
    target_cost_per_token: Optional[float] = None
    target_tokens_per_joule: Optional[float] = None
    target_gpu_utilization_pct: Optional[float] = None
    preferred_carbon_intensity: Optional[float] = None  # gCO2/kWh
    preferred_energy_price_percentile: Optional[float] = None  # 0..100
    preferred_latency_headroom_pct: Optional[float] = None  # 0..100
    max_acceptable_savings_tradeoff_pct: Optional[float] = None  # 0..100
    optimization_aggressiveness: Optional[OptimizationAggressiveness] = None

    def validate(self, prefix: str = "soft") -> list[str]:
        errs: list[str] = []

        def _pct(name: str, val: Optional[float]):
            if val is not None and not (0.0 <= float(val) <= 100.0):
                errs.append(f"{prefix}.{name} must be between 0 and 100, got {val}")

        def _nonneg(name: str, val):
            if val is not None and float(val) < 0:
                errs.append(f"{prefix}.{name} must be >= 0, got {val}")

        _nonneg("target_cost_per_token", self.target_cost_per_token)
        _nonneg("target_tokens_per_joule", self.target_tokens_per_joule)
        _pct("target_gpu_utilization_pct", self.target_gpu_utilization_pct)
        _nonneg("preferred_carbon_intensity", self.preferred_carbon_intensity)
        _pct("preferred_energy_price_percentile", self.preferred_energy_price_percentile)
        _pct("preferred_latency_headroom_pct", self.preferred_latency_headroom_pct)
        _pct("max_acceptable_savings_tradeoff_pct", self.max_acceptable_savings_tradeoff_pct)

        if self.preferred_regions is not None and not isinstance(self.preferred_regions, list):
            errs.append(f"{prefix}.preferred_regions must be a list of region strings")

        if self.optimization_aggressiveness is not None and not isinstance(
            self.optimization_aggressiveness, OptimizationAggressiveness
        ):
            errs.append(
                f"{prefix}.optimization_aggressiveness must be one of "
                f"{[a.value for a in OptimizationAggressiveness]}"
            )
        return errs


@dataclass
class SLAPolicy:
    """A complete SLA policy attaching hard + soft constraints to a workload.

    Attributes:
        name: Human-readable policy name (also used as registry key fallback).
        tier: Priority tier. Supplies defaults for unspecified fields.
        hard: Hard constraints (blocking).
        soft: Soft constraints (penalizing).
        applies_to_workloads: Explicit workload/service IDs this policy covers.
        applies_to_workload_types: Workload-type names this policy covers.
        enabled: Per-policy enforcement switch. When False the policy is
            ingested and reported but never blocks/penalizes (audit-only).
        description: Free-text description.
    """

    name: str
    tier: PriorityTier = PriorityTier.STANDARD
    hard: HardSLA = field(default_factory=HardSLA)
    soft: SoftSLA = field(default_factory=SoftSLA)
    applies_to_workloads: list[str] = field(default_factory=list)
    applies_to_workload_types: list[str] = field(default_factory=list)
    enabled: bool = True
    description: str = ""

    @property
    def aggressiveness(self) -> OptimizationAggressiveness:
        """Effective aggressiveness (soft override, else tier default)."""
        if self.soft.optimization_aggressiveness is not None:
            return self.soft.optimization_aggressiveness
        return TIER_DEFAULTS[self.tier].soft.optimization_aggressiveness  # type: ignore[return-value]

    def validate(self) -> list[str]:
        errs: list[str] = []
        if not self.name or not isinstance(self.name, str):
            errs.append("policy.name is required and must be a non-empty string")
        if not isinstance(self.tier, PriorityTier):
            errs.append(
                f"policy.tier must be one of {[t.value for t in PriorityTier]}, got {self.tier!r}"
            )
        errs.extend(self.hard.validate())
        errs.extend(self.soft.validate())
        return errs


# ---------------------------------------------------------------------------
# Tier defaults: safest (critical) -> most cost-optimized (batch).
# These are MERGED UNDER an explicit policy — explicit fields always win.
# ---------------------------------------------------------------------------
def _tier_defaults() -> dict[PriorityTier, SLAPolicy]:
    return {
        PriorityTier.CRITICAL: SLAPolicy(
            name="tier:critical",
            tier=PriorityTier.CRITICAL,
            hard=HardSLA(
                max_p95_latency_ms=200.0,
                max_p99_latency_ms=500.0,
                max_queue_wait_ms=100.0,
                min_availability_pct=99.95,
                max_error_rate_pct=0.1,
                max_timeout_rate_pct=0.1,
                migration_allowed=False,
                max_migrations_per_hour=0,
                required_capacity_buffer_pct=30.0,
            ),
            soft=SoftSLA(
                preferred_latency_headroom_pct=40.0,
                max_acceptable_savings_tradeoff_pct=2.0,
                optimization_aggressiveness=OptimizationAggressiveness.CONSERVATIVE,
            ),
            description="Critical: safest. No risky migrations, large capacity buffer.",
        ),
        PriorityTier.LATENCY_SENSITIVE: SLAPolicy(
            name="tier:latency_sensitive",
            tier=PriorityTier.LATENCY_SENSITIVE,
            hard=HardSLA(
                max_p95_latency_ms=400.0,
                max_p99_latency_ms=1000.0,
                max_queue_wait_ms=500.0,
                min_availability_pct=99.9,
                max_error_rate_pct=0.5,
                max_timeout_rate_pct=0.5,
                migration_allowed=True,
                max_migrations_per_hour=1,
                required_capacity_buffer_pct=20.0,
            ),
            soft=SoftSLA(
                preferred_latency_headroom_pct=30.0,
                max_acceptable_savings_tradeoff_pct=5.0,
                optimization_aggressiveness=OptimizationAggressiveness.CONSERVATIVE,
            ),
            description="Latency-sensitive: tight latency, migrations rate-limited.",
        ),
        PriorityTier.STANDARD: SLAPolicy(
            name="tier:standard",
            tier=PriorityTier.STANDARD,
            hard=HardSLA(
                max_p95_latency_ms=1000.0,
                max_p99_latency_ms=3000.0,
                max_queue_wait_ms=2000.0,
                min_availability_pct=99.5,
                max_error_rate_pct=1.0,
                max_timeout_rate_pct=1.0,
                migration_allowed=True,
                max_migrations_per_hour=2,
                required_capacity_buffer_pct=10.0,
            ),
            soft=SoftSLA(
                preferred_latency_headroom_pct=20.0,
                max_acceptable_savings_tradeoff_pct=10.0,
                optimization_aggressiveness=OptimizationAggressiveness.BALANCED,
            ),
            description="Standard: balanced cost vs SLA risk.",
        ),
        PriorityTier.FLEXIBLE: SLAPolicy(
            name="tier:flexible",
            tier=PriorityTier.FLEXIBLE,
            hard=HardSLA(
                max_p99_latency_ms=10000.0,
                max_queue_wait_ms=30000.0,
                min_availability_pct=99.0,
                max_error_rate_pct=3.0,
                max_timeout_rate_pct=3.0,
                migration_allowed=True,
                max_migrations_per_hour=6,
                required_capacity_buffer_pct=5.0,
            ),
            soft=SoftSLA(
                preferred_latency_headroom_pct=10.0,
                max_acceptable_savings_tradeoff_pct=20.0,
                optimization_aggressiveness=OptimizationAggressiveness.BALANCED,
            ),
            description="Flexible: tolerates moderate SLA risk for savings.",
        ),
        PriorityTier.BATCH: SLAPolicy(
            name="tier:batch",
            tier=PriorityTier.BATCH,
            hard=HardSLA(
                # Batch tolerates very high latency/queue; only reliability floors.
                max_queue_wait_ms=3_600_000.0,  # up to an hour of queueing OK
                min_availability_pct=95.0,
                max_error_rate_pct=5.0,
                max_timeout_rate_pct=10.0,
                migration_allowed=True,
                max_migrations_per_hour=60,
                required_capacity_buffer_pct=0.0,
            ),
            soft=SoftSLA(
                preferred_latency_headroom_pct=0.0,
                max_acceptable_savings_tradeoff_pct=50.0,
                optimization_aggressiveness=OptimizationAggressiveness.AGGRESSIVE,
            ),
            description="Batch: most cost-optimized. Aggressive migration allowed.",
        ),
    }


TIER_DEFAULTS: dict[PriorityTier, SLAPolicy] = _tier_defaults()


def _merge_dataclass(base, override):
    """Return a new instance of the dataclass type filling base fields with
    override's non-None values. ``override`` wins where it specifies a value."""
    merged_kwargs = {}
    for f in fields(base):
        ov = getattr(override, f.name)
        bv = getattr(base, f.name)
        merged_kwargs[f.name] = ov if ov is not None else bv
    return type(base)(**merged_kwargs)


def apply_tier_defaults(policy: SLAPolicy) -> SLAPolicy:
    """Return a copy of *policy* with unspecified hard/soft fields filled from
    the policy's priority-tier defaults.

    Explicit (non-None) fields on the policy ALWAYS win. This is how
    ``critical = safest`` and ``batch = most cost-optimized`` defaults take
    effect without the user having to restate every field.
    """
    tier_default = TIER_DEFAULTS[policy.tier]
    merged_hard = _merge_dataclass(tier_default.hard, policy.hard)
    merged_soft = _merge_dataclass(tier_default.soft, policy.soft)
    return SLAPolicy(
        name=policy.name,
        tier=policy.tier,
        hard=merged_hard,
        soft=merged_soft,
        applies_to_workloads=list(policy.applies_to_workloads),
        applies_to_workload_types=list(policy.applies_to_workload_types),
        enabled=policy.enabled,
        description=policy.description,
    )
