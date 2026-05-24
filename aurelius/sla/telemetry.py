"""Telemetry interfaces + conservative heuristic predictor for SLA evaluation.

The SLA correction engine reasons over two snapshots:

  * ``WorkloadState`` — the CURRENT observed/estimated state of a workload in
    its current region.
  * a PREDICTED ``WorkloadState`` — what the state is expected to look like
    AFTER a candidate action executes.

Real deployments should feed observed telemetry (Prometheus/DCGM, queue
provider, latency histograms) into these snapshots. Where a prediction model
does not yet exist, :class:`HeuristicPredictor` provides a CONSERVATIVE,
clearly-documented estimate. It is deliberately pessimistic (assumes risk goes
up under disruptive actions) so the SLA gate fails safe.

WHAT IS REAL vs PLACEHOLDER
---------------------------
* ``WorkloadState`` / ``RegionContext`` are plain data carriers — real if you
  populate them from real telemetry.
* ``HeuristicPredictor`` is a PLACEHOLDER physics-free heuristic. Every
  assumption is marked with ``# HEURISTIC``/``# TODO``. It does not claim to
  predict latency accurately; it produces conservative upper-bound-ish deltas
  so the gate does not approve risky moves on optimistic assumptions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Optional, Protocol

from .actions import ActionType, OptimizationAction

logger = logging.getLogger(__name__)


@dataclass
class WorkloadState:
    """Observed or predicted state of a workload at a point in time.

    All fields optional: an SLA constraint over a metric that has no telemetry
    is treated as UNKNOWN. The evaluator's policy for unknown metrics is
    configurable (default: do not block on unknowns, but flag in explanation),
    so Aurelius never claims an SLA is met on data it does not have.
    """

    region: Optional[str] = None
    timestamp: Optional[datetime] = None

    # Latency (ms)
    p95_latency_ms: Optional[float] = None
    p99_latency_ms: Optional[float] = None

    # Queue
    queue_depth: Optional[float] = None
    queue_wait_ms: Optional[float] = None

    # Utilization / reliability
    gpu_utilization_pct: Optional[float] = None
    error_rate_pct: Optional[float] = None
    timeout_rate_pct: Optional[float] = None
    availability_pct: Optional[float] = None

    # Capacity / migration accounting
    capacity_buffer_pct: Optional[float] = None
    migration_count_last_hour: int = 0

    # Energy / carbon context
    energy_price: Optional[float] = None  # $/MWh
    carbon_intensity: Optional[float] = None  # gCO2/kWh
    energy_price_percentile: Optional[float] = None  # 0..100 vs region history

    # Efficiency (optional, used for soft targets)
    cost_per_token: Optional[float] = None
    tokens_per_joule: Optional[float] = None

    def copy(self) -> "WorkloadState":
        return replace(self)


@dataclass
class RegionContext:
    """Static-ish context about a candidate region used by the predictor.

    Populate from a region registry / capacity planner. Defaults are neutral.
    """

    region: str
    spare_capacity_pct: float = 50.0  # how much headroom the region has now
    baseline_p99_latency_ms: Optional[float] = None
    baseline_queue_wait_ms: Optional[float] = None
    thermally_stressed: bool = False  # PDU/cooling near limit
    throttling: bool = False  # GPUs currently clock-throttling
    network_rtt_ms: float = 0.0  # added serving latency vs current region
    energy_price: Optional[float] = None
    carbon_intensity: Optional[float] = None
    energy_price_percentile: Optional[float] = None


class TelemetryProvider(Protocol):
    """Interface a real deployment implements to supply live telemetry.

    Aurelius core ships data carriers + a heuristic; integrating a real
    metrics backend means implementing this Protocol. No fake production
    integration is bundled.
    """

    def current_state(self, workload_id: str) -> WorkloadState: ...

    def region_context(self, region: str) -> RegionContext: ...


@dataclass
class StaticTelemetryProvider:
    """A simple in-memory provider — useful for tests, backtests, and the CLI.

    Real deployments swap this for a Prometheus/DCGM-backed implementation.
    """

    states: dict[str, WorkloadState] = field(default_factory=dict)
    regions: dict[str, RegionContext] = field(default_factory=dict)

    def current_state(self, workload_id: str) -> WorkloadState:
        return self.states.get(workload_id, WorkloadState())

    def region_context(self, region: str) -> RegionContext:
        return self.regions.get(region, RegionContext(region=region))


class HeuristicPredictor:
    """Conservative, clearly-marked heuristic predictor.

    NOT a trained model. Produces a predicted :class:`WorkloadState` for a
    candidate action by applying pessimistic deltas to the current state and
    the destination region context. Intent: never let the SLA gate approve a
    risky action on optimistic assumptions.

    The numeric deltas below are HEURISTIC constants. Replace with a learned
    predictor when telemetry history is available (TODOs marked).
    """

    # HEURISTIC penalty knobs (multipliers / additive ms). Tunable, not learned.
    MIGRATION_P99_INFLATION = 1.25  # # HEURISTIC: migration adds 25% p99 tail risk transiently
    MIGRATION_COLD_START_QUEUE_MS = 5000.0  # # HEURISTIC: cold start adds queue/TTFT wait
    MIGRATION_AVAILABILITY_DROP_PCT = 0.2  # # HEURISTIC: brief availability dip during cutover
    CONSOLIDATE_UTIL_GAIN_PCT = 25.0  # # HEURISTIC: consolidation raises utilization
    CONSOLIDATE_QUEUE_INFLATION = 1.5  # # HEURISTIC: consolidation lengthens queues
    CONSOLIDATE_P99_INFLATION = 1.3  # # HEURISTIC: contention raises p99
    SPREAD_QUEUE_RELIEF = 0.6  # # HEURISTIC: spreading shortens queues
    LOW_SPARE_CAPACITY_THRESHOLD_PCT = 20.0  # # HEURISTIC: below this, risk rises
    THERMAL_P99_INFLATION = 1.4  # # HEURISTIC: thermal stress / throttle raises tail latency

    def predict(
        self,
        action: OptimizationAction,
        current: WorkloadState,
        dest_region_ctx: Optional[RegionContext] = None,
    ) -> WorkloadState:
        """Return a predicted post-action WorkloadState (conservative)."""
        pred = current.copy()
        at = action.action_type

        # Target region defaults to current region.
        pred.region = action.target_region or current.region

        # Apply region-context-driven serving latency + energy/carbon swap.
        if dest_region_ctx is not None and action.target_region:
            pred.energy_price = dest_region_ctx.energy_price if dest_region_ctx.energy_price is not None else pred.energy_price
            pred.carbon_intensity = (
                dest_region_ctx.carbon_intensity
                if dest_region_ctx.carbon_intensity is not None
                else pred.carbon_intensity
            )
            pred.energy_price_percentile = (
                dest_region_ctx.energy_price_percentile
                if dest_region_ctx.energy_price_percentile is not None
                else pred.energy_price_percentile
            )
            # Destination baseline latency/queue becomes the new floor.
            base_p99 = dest_region_ctx.baseline_p99_latency_ms
            if base_p99 is not None:
                # network RTT adds to serving latency for the remote region.
                base_p99 = base_p99 + dest_region_ctx.network_rtt_ms
                pred.p99_latency_ms = base_p99
                if pred.p95_latency_ms is not None and current.p99_latency_ms:
                    # keep p95/p99 ratio roughly stable
                    ratio = current.p95_latency_ms / current.p99_latency_ms
                    pred.p95_latency_ms = base_p99 * ratio
                elif pred.p95_latency_ms is not None:
                    pred.p95_latency_ms = base_p99 * 0.7  # # HEURISTIC ratio
            if dest_region_ctx.baseline_queue_wait_ms is not None:
                pred.queue_wait_ms = dest_region_ctx.baseline_queue_wait_ms

        # --- Action-specific conservative deltas ---
        if at in (
            ActionType.MIGRATE,
            ActionType.REROUTE,
            ActionType.CHOOSE_CHEAPER_REGION,
            ActionType.CHOOSE_LOWER_CARBON_REGION,
            ActionType.CHANGE_PLACEMENT,
        ):
            # Migration / cold start inflates p99 and adds queue wait transiently.
            if pred.p99_latency_ms is not None:
                pred.p99_latency_ms *= self.MIGRATION_P99_INFLATION
            if pred.p95_latency_ms is not None:
                pred.p95_latency_ms *= self.MIGRATION_P99_INFLATION
            pred.queue_wait_ms = (pred.queue_wait_ms or 0.0) + self.MIGRATION_COLD_START_QUEUE_MS
            if pred.availability_pct is not None:
                pred.availability_pct = max(
                    0.0, pred.availability_pct - self.MIGRATION_AVAILABILITY_DROP_PCT
                )
            pred.migration_count_last_hour = current.migration_count_last_hour + 1

            # Moving to a low-spare-capacity region increases queue + p99 risk.
            if dest_region_ctx is not None:
                if dest_region_ctx.spare_capacity_pct < self.LOW_SPARE_CAPACITY_THRESHOLD_PCT:
                    if pred.queue_wait_ms is not None:
                        pred.queue_wait_ms *= 1.5  # # HEURISTIC
                    if pred.p99_latency_ms is not None:
                        pred.p99_latency_ms *= 1.2  # # HEURISTIC
                    # capacity buffer at destination is its spare capacity
                    pred.capacity_buffer_pct = dest_region_ctx.spare_capacity_pct
                else:
                    pred.capacity_buffer_pct = dest_region_ctx.spare_capacity_pct
                # Thermal / throttling destination inflates tail latency.
                if dest_region_ctx.thermally_stressed or dest_region_ctx.throttling:
                    if pred.p99_latency_ms is not None:
                        pred.p99_latency_ms *= self.THERMAL_P99_INFLATION
                    if pred.p95_latency_ms is not None:
                        pred.p95_latency_ms *= self.THERMAL_P99_INFLATION

        elif at == ActionType.CONSOLIDATE:
            if pred.gpu_utilization_pct is not None:
                pred.gpu_utilization_pct = min(
                    100.0, pred.gpu_utilization_pct + self.CONSOLIDATE_UTIL_GAIN_PCT
                )
            if pred.queue_wait_ms is not None:
                pred.queue_wait_ms *= self.CONSOLIDATE_QUEUE_INFLATION
            else:
                pred.queue_wait_ms = self.MIGRATION_COLD_START_QUEUE_MS
            if pred.p99_latency_ms is not None:
                pred.p99_latency_ms *= self.CONSOLIDATE_P99_INFLATION
            if pred.p95_latency_ms is not None:
                pred.p95_latency_ms *= self.CONSOLIDATE_P99_INFLATION
            # Consolidation reduces the capacity buffer.
            if pred.capacity_buffer_pct is not None:
                pred.capacity_buffer_pct = max(0.0, pred.capacity_buffer_pct - self.CONSOLIDATE_UTIL_GAIN_PCT)

        elif at == ActionType.SPREAD:
            if pred.queue_wait_ms is not None:
                pred.queue_wait_ms *= self.SPREAD_QUEUE_RELIEF
            if pred.gpu_utilization_pct is not None:
                pred.gpu_utilization_pct = max(0.0, pred.gpu_utilization_pct - 15.0)  # # HEURISTIC

        elif at == ActionType.SCALE_REPLICAS:
            if action.target_replicas is not None and pred.queue_wait_ms is not None:
                # More replicas relieve queue; fewer inflate it (rough inverse).
                # TODO: replace with M/M/c queueing model when arrival rate known.
                pred.queue_wait_ms *= 0.7 if action.target_replicas > 0 else 1.3  # # HEURISTIC

        elif at == ActionType.DEFER:
            # Deferring adds queue wait equal to the defer horizon (if provided).
            defer_ms = float(action.metadata.get("defer_ms", 0.0))
            pred.queue_wait_ms = (pred.queue_wait_ms or 0.0) + defer_ms

        # KEEP: no change.
        return pred
