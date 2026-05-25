"""Binding constraint classifier for Aurelius constraint-aware GPU orchestration.

Design rules:
- Scores are normalized [0, 1]; None means "required signals absent" (not zero).
- A None score excludes the family from binding-constraint selection.
- binding_constraint is None when no family clears the confidence floor, or when
  required signals for all families are absent.
- Hysteresis: a family must be the highest-scoring for N consecutive snapshots
  (configurable, default 2) before it becomes the binding constraint.
- Tie-break: SLA-risk-reducing constraints (LATENCY, MEMORY, COMMUNICATION,
  THERMAL) beat cost constraints (ENERGY, QUEUE, UTILIZATION, TOPOLOGY) when
  scores are within the tie_margin.
- Confidence: signal_completeness × staleness_weight × provenance_weight.
  Missing signals reduce confidence; sandbox provenance is usable for sim
  validation but is_sandbox=True is passed through for downstream rejection.
- All thresholds are in ConstraintConfig and marked # HEURISTIC so they can
  be tuned without hiding their nature as engineering guesses.

This module is read-only over ClusterState. It never calls the optimizer,
execution adapters, or any inference/runtime internals.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from ..sla.actions import ActionType
from ..state.models import (
    ClusterState,
    ConstraintAssessment,
    ConstraintType,
    Provenance,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Safe / disallowed action tables (per §8 of the plan)
# ---------------------------------------------------------------------------

# Only ActionType values that are meaningful at the orchestration level are
# listed here. Runtime/inference-plane restrictions (no NCCL tuning, no KV
# cache internals) are enforced at the recommendation engine layer (Phase 9).

_SAFE_ACTIONS: dict[ConstraintType, list[str]] = {
    ConstraintType.ENERGY: [
        ActionType.DEFER.value,
        ActionType.CHOOSE_CHEAPER_REGION.value,
        ActionType.MIGRATE.value,
        ActionType.CHOOSE_LOWER_CARBON_REGION.value,
    ],
    ConstraintType.THERMAL: [
        ActionType.SPREAD.value,
        ActionType.REROUTE.value,
        ActionType.DEFER.value,
    ],
    ConstraintType.QUEUE: [
        ActionType.REROUTE.value,
        ActionType.SPREAD.value,
        ActionType.SCALE_REPLICAS.value,
        ActionType.DEFER.value,
    ],
    ConstraintType.LATENCY: [
        ActionType.SPREAD.value,
        ActionType.SCALE_REPLICAS.value,
        ActionType.REROUTE.value,
    ],
    ConstraintType.COMMUNICATION: [
        ActionType.CHANGE_PLACEMENT.value,
        ActionType.MIGRATE.value,
        ActionType.SPREAD.value,
    ],
    ConstraintType.MEMORY: [
        ActionType.SPREAD.value,
        ActionType.REROUTE.value,
        ActionType.DEFER.value,
        ActionType.SCALE_REPLICAS.value,
    ],
    ConstraintType.TOPOLOGY: [
        ActionType.CHANGE_PLACEMENT.value,
        ActionType.CONSOLIDATE.value,   # topology-aware bin-packing onto NVLink
    ],
    ConstraintType.UTILIZATION: [
        ActionType.CONSOLIDATE.value,
        ActionType.DEFER.value,
        ActionType.SPREAD.value,
    ],
    ConstraintType.NONE: [
        ActionType.KEEP.value,
    ],
}

_DISALLOWED_ACTIONS: dict[ConstraintType, list[str]] = {
    ConstraintType.ENERGY: [],         # SLA evaluator gates latency-critical migration
    ConstraintType.THERMAL: [
        ActionType.CONSOLIDATE.value,  # would increase thermal density
    ],
    ConstraintType.QUEUE: [
        ActionType.CONSOLIDATE.value,  # reduces capacity during surge
    ],
    ConstraintType.LATENCY: [
        ActionType.MIGRATE.value,      # cold-start p99 tail risk for the hot workload
    ],
    ConstraintType.COMMUNICATION: [],  # runtime disallowances enforced at engine level
    ConstraintType.MEMORY: [],         # KV cache disallowances enforced at engine level
    ConstraintType.TOPOLOGY: [],
    ConstraintType.UTILIZATION: [],
    ConstraintType.NONE: [],
}

# Tie-break priority (lower = higher priority). When two families score within
# tie_margin of each other, the lower-numbered one wins.
# SLA-risk-reducing families beat cost/efficiency families.
_TIEBREAK_PRIORITY: dict[ConstraintType, int] = {
    ConstraintType.LATENCY: 0,       # SLA-risk
    ConstraintType.MEMORY: 1,        # SLA-risk (indirect memory / cache pressure)
    ConstraintType.COMMUNICATION: 2, # SLA-risk (effective throughput)
    ConstraintType.THERMAL: 3,       # SLA-risk (throttle → latency spike)
    ConstraintType.ENERGY: 4,        # cost optimization
    ConstraintType.QUEUE: 5,         # operational efficiency
    ConstraintType.UTILIZATION: 6,   # cost/efficiency
    ConstraintType.TOPOLOGY: 7,      # placement optimization
    ConstraintType.NONE: 99,
}

# Number of required signals expected per family. Used for signal-completeness
# scoring even when a family cannot be scored due to absent data.
_EXPECTED_SIGNALS_PER_FAMILY = 2  # HEURISTIC: rough expectation per family


# ---------------------------------------------------------------------------
# Classifier configuration
# ---------------------------------------------------------------------------

@dataclass
class ConstraintConfig:
    """Configurable thresholds for the constraint classifier.

    Every field here is a HEURISTIC engineering guess calibrated on synthetic
    cluster scenarios. They should be tuned against real telemetry before any
    production savings/SLA claim.
    """

    # Binding detection
    binding_threshold: float = 0.35      # HEURISTIC: minimum score to be "binding"
    confidence_floor: float = 0.25       # HEURISTIC: minimum confidence to emit binding_constraint
    hysteresis_count: int = 2            # HEURISTIC: N consecutive snapshots required to flip
    tie_margin: float = 0.05             # HEURISTIC: within this, apply tie-break priority

    # Staleness
    max_acceptable_age_s: float = 300.0  # HEURISTIC: 5 min; older → staleness penalty

    # Energy
    energy_high_pct_threshold: float = 70.0   # HEURISTIC: price percentile → significant
    energy_spread_ref: float = 0.30            # HEURISTIC: (max-min)/mean spread ratio = max score

    # Thermal
    thermal_safe_c: float = 70.0         # HEURISTIC: below = no thermal pressure
    thermal_crit_c: float = 95.0         # HEURISTIC: at this temp = max score
    # Bitmask constants (DCGM CLOCKS_EVENT_REASONS thermal bits)
    _thermal_bits: int = field(default=0x20 | 0x40 | 0x80 | 0x04 | 0x08, init=False, repr=False)

    # Queue
    queue_wait_ref_ms: float = 500.0     # HEURISTIC: p95 queue wait above this = high pressure
    queue_depth_ref: float = 50.0        # HEURISTIC: reference queue depth for scoring

    # Latency
    default_sla_p99_ms: float = 2000.0   # HEURISTIC: TTFT P99 SLA when none configured
    latency_e2e_sla_ms: float = 30000.0  # HEURISTIC: end-to-end p99 SLA (TTFT + TPOT × tokens)
    latency_warning_ratio: float = 0.5   # HEURISTIC: score starts when latency > 50% of SLA

    # Memory
    mem_high_threshold: float = 0.80     # HEURISTIC: GPU HBM usage fraction above which to score
    kv_cache_high_threshold: float = 0.80  # HEURISTIC: KV cache usage above which to score

    # Utilization (under-utilization detection)
    util_low_threshold: float = 35.0     # HEURISTIC: GPU util below this = underutilization waste
    util_target: float = 70.0           # HEURISTIC: target GPU utilization

    # Communication
    # Raw bytes are hard to normalize; use a proxy:
    # score if (nvlink+pcie) tx/s > this threshold (bytes/s), combined with low SM
    comm_bytes_high_threshold: float = 5e9   # HEURISTIC: 5 GB/s tx combined = noteworthy
    comm_low_sm_threshold: float = 0.50      # HEURISTIC: SM activity below this with high comm = stalled

    # Topology (fragmentation)
    topology_poor_classes: frozenset = field(
        default_factory=lambda: frozenset({"pcie", "cross_numa", "unknown"}),
        repr=False,
    )


# ---------------------------------------------------------------------------
# Per-family scorer functions
# ---------------------------------------------------------------------------

def _score_energy(state: ClusterState, cfg: ConstraintConfig) -> tuple[Optional[float], list[str]]:
    """Score energy-bound pressure. Returns (score, missing_signals)."""
    missing: list[str] = []
    prices: list[float] = []
    pct_scores: list[float] = []
    power_cap_scores: list[float] = []

    for region_id, region in state.regions.items():
        if region.energy is None:
            missing.append(f"energy[{region_id}]")
            continue
        e = region.energy
        if e.price_per_mwh is not None:
            prices.append(e.price_per_mwh)
        else:
            missing.append(f"energy.price_per_mwh[{region_id}]")

        if e.price_percentile is not None:
            pct_scores.append(e.price_percentile / 100.0)

        if e.power_cap_kw is not None and e.power_draw_kw is not None and e.power_cap_kw > 0:
            power_cap_scores.append(e.power_draw_kw / e.power_cap_kw)

    if not prices and not pct_scores:
        return None, missing

    # Cross-region spread: high when max region is much more expensive than min
    spread_score = 0.0
    if len(prices) >= 2:
        price_range = max(prices) - min(prices)
        mean_price = sum(prices) / len(prices)
        if mean_price > 0:
            spread_ratio = price_range / mean_price
            spread_score = min(1.0, spread_ratio / cfg.energy_spread_ref)

    # Percentile: how expensive is the current moment vs history?
    pct_score = 0.0
    if pct_scores:
        pct_score = max(pct_scores)
        # Only score above the threshold
        pct_score = max(0.0, (pct_score - cfg.energy_high_pct_threshold / 100.0) /
                        (1.0 - cfg.energy_high_pct_threshold / 100.0))
    elif prices:
        # No percentile history — use cross-region spread only with reduced weight
        pass

    # Power cap proximity
    cap_score = 0.0
    if power_cap_scores:
        cap_score = max(power_cap_scores)
        cap_score = max(0.0, (cap_score - 0.80) / 0.20)  # HEURISTIC: 80%→100% of cap

    # Weighted combination
    if pct_scores:
        score = 0.55 * pct_score + 0.30 * spread_score + 0.15 * min(1.0, cap_score)
    else:
        # No percentile: rely on spread + cap
        score = 0.60 * spread_score + 0.40 * min(1.0, cap_score)

    return min(1.0, max(0.0, score)), missing


def _score_thermal(state: ClusterState, cfg: ConstraintConfig) -> tuple[Optional[float], list[str]]:
    """Score thermal-bound pressure. Returns (score, missing_signals)."""
    missing: list[str] = []
    temps: list[float] = []
    throttle_states: list[bool] = []

    for region in state.regions.values():
        for node in region.nodes.values():
            for gpu in node.gpus.values():
                if gpu.temp_c is not None:
                    temps.append(gpu.temp_c)
                else:
                    missing.append(f"gpu.temp_c[{gpu.gpu_uuid[:8]}]")

                if gpu.clocks_event_reasons is not None:
                    throttle_states.append(gpu.thermal_throttling or False)
                # If clocks_event_reasons is None, we simply don't know throttle state

    # Check ThermalState if available (aggregated view)
    for region in state.regions.values():
        if region.thermal is not None:
            th = region.thermal
            if th.max_gpu_temp_c is not None and not temps:
                temps.append(th.max_gpu_temp_c)
            if th.throttling_fraction is not None and not throttle_states:
                throttle_states.append(th.throttling_fraction > 0)

    if not temps:
        missing.append("gpu.temp_c")
        return None, missing

    max_temp = max(temps)
    temp_range = cfg.thermal_crit_c - cfg.thermal_safe_c
    if temp_range <= 0:
        temp_score = 0.0
    else:
        temp_score = max(0.0, min(1.0, (max_temp - cfg.thermal_safe_c) / temp_range))

    if throttle_states:
        throttle_fraction = sum(1 for t in throttle_states if t) / len(throttle_states)
    else:
        throttle_fraction = 0.0

    score = 0.65 * temp_score + 0.35 * throttle_fraction
    return min(1.0, max(0.0, score)), missing


def _score_queue(state: ClusterState, cfg: ConstraintConfig) -> tuple[Optional[float], list[str]]:
    """Score queue-bound pressure. Returns (score, missing_signals)."""
    missing: list[str] = []
    queue_scores: list[float] = []
    has_any_signal = False

    for service in state.all_services.values():
        if service.requests_waiting is not None:
            has_any_signal = True
            depth_score = min(1.0, service.requests_waiting / cfg.queue_depth_ref)
            queue_scores.append(depth_score)
        else:
            missing.append(f"service.requests_waiting[{service.service_id}]")

        if service.queue_time_p95_ms is not None:
            has_any_signal = True
            wait_score = min(1.0, service.queue_time_p95_ms / cfg.queue_wait_ref_ms)
            queue_scores.append(wait_score)

    # Also check spare capacity across regions — low spare + high queue = doubly bound
    spare_scores: list[float] = []
    for region in state.regions.values():
        if region.spare_capacity_pct is not None:
            has_any_signal = True
            # Low spare → high pressure
            spare_score = max(0.0, 1.0 - region.spare_capacity_pct / 100.0)
            spare_scores.append(spare_score)

    if not has_any_signal:
        missing.append("requests_waiting, queue_time, spare_capacity")
        return None, missing

    base_score = max(queue_scores) if queue_scores else 0.0
    spare_score = max(spare_scores) if spare_scores else 0.0

    # Combine: queue depth + spare capacity constraint
    score = 0.70 * base_score + 0.30 * spare_score
    return min(1.0, max(0.0, score)), missing


def _score_latency(state: ClusterState, cfg: ConstraintConfig) -> tuple[Optional[float], list[str]]:
    """Score latency-bound pressure. Returns (score, missing_signals)."""
    missing: list[str] = []
    headrooms: list[float] = []
    has_any_signal = False

    def _headroom(actual: Optional[float], sla_limit: float) -> Optional[float]:
        if actual is None:
            return None
        ratio = actual / max(sla_limit, 1.0)
        # 0 below warning_ratio of SLA, 1 at or above SLA limit
        return max(0.0, min(1.0, (ratio - cfg.latency_warning_ratio) /
                            (1.0 - cfg.latency_warning_ratio)))

    for service in state.all_services.values():
        # ttft_p99_ms is compared against the tight TTFT SLA (default 2000ms).
        # p99_latency_ms and p95_latency_ms are end-to-end metrics that include
        # token-generation time (TTFT + TPOT × output_tokens). For LLM inference,
        # end-to-end p99 can legitimately be 10–30 s, so we use a much higher
        # reference (latency_e2e_sla_ms, default 30 000 ms) to avoid false positives.
        sla_ttft = cfg.default_sla_p99_ms
        sla_e2e = cfg.latency_e2e_sla_ms

        h = _headroom(service.ttft_p99_ms, sla_ttft)
        if h is not None:
            has_any_signal = True
            headrooms.append(h)

        h = _headroom(service.p99_latency_ms, sla_e2e)
        if h is not None:
            has_any_signal = True
            headrooms.append(h)
        elif service.p99_latency_ms is None and service.ttft_p99_ms is None:
            missing.append(f"service.p99_latency_ms[{service.service_id}]")

        h = _headroom(service.p95_latency_ms, sla_e2e * 0.7)
        if h is not None:
            has_any_signal = True
            headrooms.append(h)

    if not has_any_signal:
        missing.append("p95/p99_latency_ms, ttft_ms")
        return None, missing

    score = max(headrooms) if headrooms else 0.0
    return min(1.0, max(0.0, score)), missing


def _score_memory(state: ClusterState, cfg: ConstraintConfig) -> tuple[Optional[float], list[str]]:
    """Score memory-bound (indirect) pressure. Returns (score, missing_signals).

    Indirect because Aurelius never manages KV cache internals. This score
    reflects observable memory pressure from HBM usage + KV cache metrics.
    """
    missing: list[str] = []
    mem_scores: list[float] = []
    has_any_signal = False

    for gpu in state.all_gpus.values():
        mem_util = gpu.mem_util_pct  # derived property: used / total (0–100)
        if mem_util is not None:
            has_any_signal = True
            # Score rises above threshold
            frac = mem_util / 100.0
            score = max(0.0, (frac - cfg.mem_high_threshold) /
                        (1.0 - cfg.mem_high_threshold))
            mem_scores.append(min(1.0, score))
        else:
            missing.append(f"gpu.mem_util[{gpu.gpu_uuid[:8]}]")

    for service in state.all_services.values():
        if service.kv_cache_usage is not None:
            has_any_signal = True
            score = max(0.0, (service.kv_cache_usage - cfg.kv_cache_high_threshold) /
                        (1.0 - cfg.kv_cache_high_threshold))
            mem_scores.append(min(1.0, score))

        if service.preemptions_total is not None and service.preemptions_total > 0:
            # Preemptions signal KV pressure even without usage metric
            has_any_signal = True
            mem_scores.append(min(1.0, service.preemptions_total / 100.0))  # HEURISTIC

    if not has_any_signal:
        missing.append("gpu.mem_util, kv_cache_usage")
        return None, missing

    score = max(mem_scores) if mem_scores else 0.0
    return min(1.0, max(0.0, score)), missing


def _score_communication(state: ClusterState, cfg: ConstraintConfig) -> tuple[Optional[float], list[str]]:
    """Score communication-bound pressure. Returns (score, missing_signals).

    Conservative: only scores if NVLink or PCIe traffic metrics are present.
    Communication-bound is the hardest to detect reliably; prefer NONE over
    a low-confidence COMMUNICATION call.
    """
    missing: list[str] = []
    comm_scores: list[float] = []
    has_any_signal = False

    for gpu in state.all_gpus.values():
        nvlink_tx = gpu.nvlink_tx_bytes_per_s
        pcie_tx = gpu.pcie_tx_bytes_per_s
        sm = gpu.sm_active_ratio

        has_comm_metric = nvlink_tx is not None or pcie_tx is not None
        if not has_comm_metric:
            continue

        has_any_signal = True
        comm_bytes = (nvlink_tx or 0.0) + (pcie_tx or 0.0)

        # Communication-bound means compute is stalled waiting on network transfers.
        # This requires BOTH high traffic AND low SM occupancy. High bytes with high
        # SM means compute is active (not stalled), so there is no constraint.
        if comm_bytes >= cfg.comm_bytes_high_threshold:
            raw = comm_bytes / cfg.comm_bytes_high_threshold
            bytes_score = min(1.0, raw - 1.0)  # 0 at threshold, 1 at 2×threshold

            if sm is not None:
                if sm < cfg.comm_low_sm_threshold:
                    # Genuine stall: high traffic + compute idle → score + amplify
                    stall_factor = 1.0 + (cfg.comm_low_sm_threshold - sm) / cfg.comm_low_sm_threshold
                    comm_scores.append(min(1.0, bytes_score * stall_factor))
                # else: SM is high → compute is not stalled; no communication pressure
            else:
                # No SM data — conservative: half-weight bytes-only score
                comm_scores.append(bytes_score * 0.5)  # HEURISTIC

    if not has_any_signal:
        missing.append("gpu.nvlink_tx_bytes_per_s, gpu.pcie_tx_bytes_per_s")
        return None, missing

    score = max(comm_scores) if comm_scores else 0.0
    return min(1.0, max(0.0, score)), missing


def _score_topology(state: ClusterState, cfg: ConstraintConfig) -> tuple[Optional[float], list[str]]:
    """Score topology-bound pressure (placement fragmentation).

    Returns (score, missing_signals). Low confidence by default — topology
    telemetry is often unavailable in cloud/managed-K8s deployments.
    """
    missing: list[str] = []
    topologies = []
    has_any_signal = False

    for region in state.regions.values():
        if region.topology is not None:
            has_any_signal = True
            topologies.append(region.topology)
        else:
            missing.append(f"topology[{region.region}]")

    if not has_any_signal:
        return None, missing

    poor_count = sum(
        1 for t in topologies
        if t.interconnect_class in cfg.topology_poor_classes
    )
    total = len(topologies)
    if total == 0:
        return 0.0, missing

    frag_fraction = poor_count / total
    return min(1.0, frag_fraction), missing


def _score_utilization(state: ClusterState, cfg: ConstraintConfig) -> tuple[Optional[float], list[str]]:
    """Score utilization-bound pressure (under-utilization / wasted capacity).

    Returns (score, missing_signals). High score = significant idle capacity.
    """
    missing: list[str] = []
    util_values: list[float] = []

    for gpu in state.all_gpus.values():
        if gpu.util_pct is not None:
            util_values.append(gpu.util_pct)
        else:
            missing.append(f"gpu.util_pct[{gpu.gpu_uuid[:8]}]")

    if not util_values:
        return None, missing

    mean_util = sum(util_values) / len(util_values)

    if mean_util >= cfg.util_target:
        # Cluster is well-utilized; not utilization-bound
        return 0.0, missing

    if mean_util < cfg.util_low_threshold:
        # Significant underutilization
        score = (cfg.util_low_threshold - mean_util) / cfg.util_low_threshold
    else:
        # Partial underutilization between low_threshold and target
        score = (cfg.util_target - mean_util) / (cfg.util_target - cfg.util_low_threshold)
        score *= 0.4  # HEURISTIC: partial underutilization scores lower

    return min(1.0, max(0.0, score)), missing


# ---------------------------------------------------------------------------
# Confidence computation
# ---------------------------------------------------------------------------

def _compute_confidence(
    state: ClusterState,
    scores: dict[ConstraintType, float],
    binding: Optional[ConstraintType],
    scored_families: int,
    total_families: int,
    cfg: ConstraintConfig,
) -> float:
    """Compute overall classifier confidence in the binding constraint diagnosis.

    Confidence reflects:
    1. How clearly is the binding constraint detected (binding score above threshold)?
    2. How fresh and trustworthy is the telemetry (staleness + provenance)?
    3. A mild penalty for very low signal coverage (scored < 2 families).

    Missing constraint families are surfaced in missing_signals — they reduce
    confidence mildly but do not invalidate a clearly-detected binding constraint.
    The operator can see the missing signals and decide how much to trust the result.

    Formula:
        binding_strength × staleness_weight × provenance_weight × coverage_factor × partial_penalty
    """
    if not scores or scored_families == 0:
        return 0.0

    # Binding strength: how clearly is the top constraint above zero?
    # Use the raw score (not normalized above threshold) so that moderate scores
    # (e.g. 0.45 for energy) still produce meaningful confidence instead of
    # collapsing to (0.45-0.35)/(1-0.35) = 0.15.
    if binding is not None:
        top_score = scores.get(binding, 0.0)
    else:
        top_score = max(scores.values()) if scores else 0.0
    binding_strength = top_score  # raw [0, 1]

    # Staleness: worst (most stale) region degrades confidence
    max_age_s = 0.0
    for region in state.regions.values():
        if region.provenance.sample_age_s is not None:
            max_age_s = max(max_age_s, region.provenance.sample_age_s)
    if max_age_s > 0 and cfg.max_acceptable_age_s > 0:
        staleness_weight = max(0.0, 1.0 - max_age_s / cfg.max_acceptable_age_s)
    else:
        staleness_weight = 1.0

    # Provenance: minimum confidence_weight across all region provenances
    prov_weights = [r.provenance.confidence_weight for r in state.regions.values()]
    prov_weight = min(prov_weights) if prov_weights else 1.0

    # Coverage factor: mild penalty when very few families are scored (< 2 of total).
    # We expect at minimum GPU util + one latency/queue signal in a real deployment.
    # HEURISTIC: 2 scored families = full coverage; 1 = 50%; 0 = 0%
    coverage_factor = min(1.0, scored_families / 2.0)

    # Partial state penalty
    partial_penalty = 0.85 if state.is_partial else 1.0

    return min(1.0, binding_strength * staleness_weight * prov_weight * coverage_factor * partial_penalty)


# ---------------------------------------------------------------------------
# Binding constraint selection with tie-break
# ---------------------------------------------------------------------------

def _select_binding(
    scores: dict[ConstraintType, float],
    cfg: ConstraintConfig,
) -> Optional[ConstraintType]:
    """Select the binding constraint, applying threshold + tie-break logic.

    Returns None if no constraint exceeds binding_threshold.
    """
    candidates = [
        (ct, s) for ct, s in scores.items()
        if s >= cfg.binding_threshold
    ]
    if not candidates:
        return None

    # Find the maximum score
    max_score = max(s for _, s in candidates)

    # Collect all within tie_margin of the max
    tied = [ct for ct, s in candidates if max_score - s <= cfg.tie_margin]

    # Among tied candidates, pick the highest-priority (lowest tiebreak number)
    tied.sort(key=lambda ct: _TIEBREAK_PRIORITY.get(ct, 99))
    return tied[0]


# ---------------------------------------------------------------------------
# Main classifier
# ---------------------------------------------------------------------------

class ConstraintClassifier:
    """Scores 8 constraint families from a ClusterState snapshot.

    Usage:
        cfg = ConstraintConfig(hysteresis_count=2)
        classifier = ConstraintClassifier(config=cfg)
        assessment = classifier.assess(cluster_state)

    The classifier is stateful: it maintains a short history of recent
    binding constraint candidates for hysteresis (anti-flapping).

    assess() is idempotent on the same ClusterState; the history only advances
    when assess() is called with new snapshots.
    """

    def __init__(self, config: Optional[ConstraintConfig] = None) -> None:
        self.config = config or ConstraintConfig()
        # Recent binding constraint candidates (pre-hysteresis)
        self._candidate_history: deque[Optional[ConstraintType]] = deque(
            maxlen=max(self.config.hysteresis_count * 2, 4)
        )
        self._stable_binding: Optional[ConstraintType] = None

    def reset(self) -> None:
        """Clear hysteresis history (e.g. when starting a new simulation run)."""
        self._candidate_history.clear()
        self._stable_binding = None

    def assess(
        self,
        state: ClusterState,
        region: Optional[str] = None,
    ) -> ConstraintAssessment:
        """Classify the binding constraint from the given ClusterState.

        Parameters
        ----------
        state:
            Canonical ClusterState snapshot from the normalization layer.
        region:
            If provided, narrow assessment to this region. None = cluster-wide.

        Returns
        -------
        ConstraintAssessment with scores, binding_constraint, confidence, and
        missing_signals. binding_constraint is None when no signal qualifies.
        """
        cfg = self.config
        ts = state.timestamp

        # Narrow to a single region if requested
        if region is not None:
            if region not in state.regions:
                return self._fail_safe(ts, state.provenance, region, f"region {region!r} not found in ClusterState")
            scoped = ClusterState(
                timestamp=state.timestamp,
                provenance=state.provenance,
                snapshot_id=state.snapshot_id,
                regions={region: state.regions[region]},
                is_partial=state.is_partial,
                missing_sources=state.missing_sources,
            )
        else:
            scoped = state

        # Run all family scorers
        scorer_results: dict[ConstraintType, tuple[Optional[float], list[str]]] = {
            ConstraintType.ENERGY: _score_energy(scoped, cfg),
            ConstraintType.THERMAL: _score_thermal(scoped, cfg),
            ConstraintType.QUEUE: _score_queue(scoped, cfg),
            ConstraintType.LATENCY: _score_latency(scoped, cfg),
            ConstraintType.MEMORY: _score_memory(scoped, cfg),
            ConstraintType.COMMUNICATION: _score_communication(scoped, cfg),
            ConstraintType.TOPOLOGY: _score_topology(scoped, cfg),
            ConstraintType.UTILIZATION: _score_utilization(scoped, cfg),
        }

        # Separate scored families (have a value) from unscored (None = absent data)
        scores: dict[ConstraintType, float] = {}
        all_missing: list[str] = []
        for ct, (score, missing) in scorer_results.items():
            if score is not None:
                scores[ct] = score
            else:
                all_missing.extend(missing)

        scored_count = len(scores)
        total_families = len(scorer_results)

        # Select candidate binding constraint (pre-hysteresis, pre-confidence-floor)
        # We need this first to compute binding-strength confidence, then re-check floor.
        candidate = _select_binding(scores, cfg)

        # Compute confidence (uses candidate binding score for binding_strength)
        confidence = _compute_confidence(scoped, scores, candidate, scored_count, total_families, cfg)

        # Apply confidence floor: suppress candidate if we're not confident enough
        if confidence < cfg.confidence_floor:
            candidate = None

        # Apply hysteresis: emit stable binding only after N consecutive identical candidates
        self._candidate_history.append(candidate)
        n = cfg.hysteresis_count
        recent = list(self._candidate_history)
        if len(recent) >= n and all(c == candidate for c in recent[-n:]):
            self._stable_binding = candidate
        # If not yet stable, keep the previous stable binding (or None)

        binding_constraint = self._stable_binding

        # Build rationale
        rationale = self._build_rationale(
            scores, binding_constraint, candidate, confidence, all_missing, n, recent
        )

        prov = Provenance(
            source="constraint-classifier",
            fetched_at=ts,
            confidence=("high" if confidence >= 0.7 else "medium" if confidence >= 0.4 else "low"),
            is_sandbox=state.provenance.is_sandbox,
        )

        safe_actions = list(_SAFE_ACTIONS.get(binding_constraint or ConstraintType.NONE, [ActionType.KEEP.value]))
        disallowed_actions = list(_DISALLOWED_ACTIONS.get(binding_constraint or ConstraintType.NONE, []))

        assessment = ConstraintAssessment(
            timestamp=ts,
            provenance=prov,
            region=region,
            scores=scores,
            binding_constraint=binding_constraint,
            confidence=round(confidence, 4),
            missing_signals=all_missing,
            rationale=rationale,
            safe_action_types=safe_actions,
            disallowed_action_types=disallowed_actions,
        )

        logger.debug(
            "ConstraintAssessment: binding=%s confidence=%.2f scored=%d/%d missing=%d",
            binding_constraint,
            confidence,
            scored_count,
            total_families,
            len(all_missing),
        )
        return assessment

    def _fail_safe(
        self,
        ts: datetime,
        provenance: Provenance,
        region: Optional[str],
        reason: str,
    ) -> ConstraintAssessment:
        """Return a safe no-op assessment when assessment cannot be completed."""
        prov = Provenance(
            source="constraint-classifier",
            fetched_at=ts,
            confidence="low",
            is_sandbox=provenance.is_sandbox,
        )
        return ConstraintAssessment(
            timestamp=ts,
            provenance=prov,
            region=region,
            scores={},
            binding_constraint=None,
            confidence=0.0,
            missing_signals=[reason],
            rationale=f"Fail-safe: {reason}. No binding constraint identified; safe action is KEEP.",
            safe_action_types=[ActionType.KEEP.value],
            disallowed_action_types=[],
        )

    @staticmethod
    def _build_rationale(
        scores: dict[ConstraintType, float],
        binding: Optional[ConstraintType],
        candidate: Optional[ConstraintType],
        confidence: float,
        missing: list[str],
        hysteresis_n: int,
        recent_history: list[Optional[ConstraintType]],
    ) -> str:
        parts: list[str] = []

        if scores:
            sorted_scores = sorted(scores.items(), key=lambda x: -x[1])
            top = ", ".join(f"{ct.value}={s:.2f}" for ct, s in sorted_scores[:4])
            parts.append(f"Scores: {top}.")

        if binding is not None:
            parts.append(f"Binding constraint: {binding.value} (confidence={confidence:.2f}).")
        elif candidate is not None:
            parts.append(
                f"Candidate {candidate.value} not yet stable "
                f"(need {hysteresis_n} consecutive; have {len(recent_history)} in history)."
            )
        else:
            parts.append("No constraint exceeds binding threshold.")

        if missing:
            shown = missing[:5]
            rest = len(missing) - len(shown)
            sig_str = ", ".join(shown) + (f" +{rest} more" if rest else "")
            parts.append(f"Missing signals: {sig_str}.")

        if confidence < 0.3:
            parts.append("Low confidence — fail-safe KEEP recommended.")

        return " ".join(parts) if parts else "No information available."
