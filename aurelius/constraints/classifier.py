"""Constraint classifier for Aurelius.

Takes a ClusterState snapshot and produces a ConstraintAssessment identifying
which constraint is currently binding the cluster.

Design principles:
- Missing signal → None score (not included), lower confidence. Never fabricated.
- All thresholds are marked # HEURISTIC — calibrate from real telemetry.
- Hysteresis prevents flapping between binding constraints.
- Each family has explicit safe/disallowed action mappings.
- No runtime or KV-cache internals are touched or observed here.

Constraint families (in ConstraintType enum order):
  ENERGY       — energy price pressure; flexible load can be deferred/moved
  THERMAL      — GPU/rack temperature; throttling active or imminent
  QUEUE        — inference queue depth or wait time is excessive
  LATENCY      — p99/TTFT latency near or above SLA budget
  COMMUNICATION — GPU-to-GPU traffic high; topology mismatch degrades throughput
  MEMORY       — HBM / KV-cache / prefix-cache pressure (indirect only)
  TOPOLOGY     — high-comm workloads placed across weak interconnect links
  UTILIZATION  — cluster underutilized / fragmented; bin-packing opportunity
  NONE         — no constraint above threshold; no action needed
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from aurelius.state.models import (
    ClusterState,
    ConstraintAssessment,
    ConstraintType,
    Provenance,
    RegionState,
)


# ---------------------------------------------------------------------------
# Per-family action lookup tables
# ---------------------------------------------------------------------------

_SAFE_ACTIONS: dict[ConstraintType, list[str]] = {
    ConstraintType.ENERGY: [
        "defer_flexible_workload",
        "shift_batch_to_cheaper_region",
        "migrate_flexible_workload",
        "reroute_traffic",
        "bin_pack_batch_workloads",
    ],
    ConstraintType.THERMAL: [
        "spread_from_hot_rack",
        "reroute_traffic",
        "preserve_affinity",
        "reduce_migration_frequency",
    ],
    ConstraintType.QUEUE: [
        "add_replica_recommendation",
        "prewarm_replica_recommendation",
        "reserve_capacity_for_sla",
        "separate_batch_from_critical",
    ],
    ConstraintType.LATENCY: [
        "preserve_affinity",
        "reduce_migration_frequency",
        "reserve_capacity_for_sla",
        "prewarm_replica_recommendation",
    ],
    ConstraintType.COMMUNICATION: [
        "colocate_communicating_workloads",
        "topology_aware_replacement",
        "topology_aware_placement",
    ],
    ConstraintType.MEMORY: [
        "preserve_cache_affinity",
        "avoid_cold_reroute",
        "prewarm_replica_recommendation",
        "add_replica_recommendation",
    ],
    ConstraintType.TOPOLOGY: [
        "topology_aware_replacement",
        "topology_aware_placement",
        "colocate_communicating_workloads",
    ],
    ConstraintType.UTILIZATION: [
        "bin_pack_batch_workloads",
        "consolidate_low_priority_workloads",
        "migrate_flexible_workload",
    ],
    ConstraintType.NONE: ["no_op"],
}

_DISALLOWED_ACTIONS: dict[ConstraintType, list[str]] = {
    ConstraintType.ENERGY: [],
    ConstraintType.THERMAL: [
        "consolidate_to_hot_rack",
        "bin_pack_onto_throttling_nodes",
    ],
    ConstraintType.QUEUE: [
        "migrate_serving_workload_during_surge",
        "reduce_replicas",
    ],
    ConstraintType.LATENCY: [
        "migrate_latency_sensitive",
        "cold_start",
        "disrupt_serving_path",
    ],
    ConstraintType.COMMUNICATION: [
        "split_communicating_workloads",
        "cross_rack_placement_for_high_comm",
    ],
    ConstraintType.MEMORY: [
        "migrate_cached_workload",
        "route_to_cold_replica",
        "modify_kv_cache_internals",
        "modify_memory_allocator",
    ],
    ConstraintType.TOPOLOGY: [
        "cross_rack_placement_for_high_comm",
    ],
    ConstraintType.UTILIZATION: [],
    ConstraintType.NONE: [],
}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ConstraintConfig:
    """Heuristic thresholds for the constraint classifier.

    All values marked # HEURISTIC — calibrate from real telemetry traces.
    Defaults are conservative starting points for common GPU inference clusters.
    """

    # --- Energy thresholds (HEURISTIC) ---
    energy_high_price_mwh: float = 100.0       # $/MWh; above → energy pressure starts
    energy_very_high_price_mwh: float = 200.0  # $/MWh; above → max energy pressure
    energy_high_percentile: float = 75.0        # 0–100; price_percentile above → pressure

    # --- Thermal thresholds (HEURISTIC) ---
    thermal_warn_temp_c: float = 80.0          # °C; GPU temp above → moderate pressure
    thermal_critical_temp_c: float = 85.0      # °C; GPU temp above → high pressure
    thermal_throttle_fraction_warn: float = 0.05  # fraction throttling → moderate pressure
    thermal_throttle_fraction_high: float = 0.20  # fraction throttling → high pressure

    # --- Queue thresholds (HEURISTIC) ---
    queue_depth_warn: float = 20.0             # requests waiting → moderate pressure
    queue_depth_high: float = 100.0            # requests waiting → high pressure
    queue_wait_warn_ms: float = 5_000.0        # ms; p99 queue wait → moderate pressure
    queue_wait_high_ms: float = 30_000.0       # ms; p99 queue wait → high pressure
    queue_oldest_warn_s: float = 60.0          # s; oldest pending age → moderate

    # --- Latency thresholds (HEURISTIC) ---
    latency_p99_warn_ms: float = 2_000.0       # ms; p99 latency → moderate pressure
    latency_p99_high_ms: float = 5_000.0       # ms; p99 latency → high pressure
    ttft_p99_warn_ms: float = 1_000.0          # ms; TTFT p99 → moderate pressure
    ttft_p99_high_ms: float = 3_000.0          # ms; TTFT p99 → high pressure
    error_rate_warn_pct: float = 1.0           # %; error rate → moderate pressure
    error_rate_high_pct: float = 5.0           # %; error rate → high pressure

    # --- Communication thresholds (HEURISTIC) ---
    nvlink_high_gbps: float = 50.0             # GB/s combined NVLink → high traffic
    nvlink_saturated_gbps: float = 200.0       # GB/s combined NVLink → saturated
    pcie_high_gbps: float = 8.0               # GB/s combined PCIe → high traffic

    # --- Memory thresholds (HEURISTIC) ---
    kv_cache_warn_fraction: float = 0.70       # kv_cache_usage above → moderate
    kv_cache_high_fraction: float = 0.85       # kv_cache_usage above → high pressure
    hbm_warn_fraction: float = 0.80            # GPU HBM usage above → moderate
    hbm_high_fraction: float = 0.90            # GPU HBM usage above → high
    prefix_hit_low_fraction: float = 0.30      # prefix_cache_hit_rate below → pressure

    # --- Utilization thresholds (HEURISTIC) ---
    util_idle_pct: float = 20.0                # GPU util below → idle
    util_low_cluster_pct: float = 40.0         # avg cluster util below → utilization pressure
    util_idle_gpu_fraction: float = 0.30       # fraction of GPUs idle → high pressure

    # --- Topology weakness scores (HEURISTIC) ---
    # Links below this score in the 0–1 ladder are considered "weak"
    # Link quality: NVSWITCH>NV4>NV3>NV2>NV1>PIX>PXB>PHB>NODE>SYS>RACK>REGION
    topology_weak_link_threshold: float = 0.4  # anything below this score is weak

    # --- Hysteresis (HEURISTIC) ---
    hysteresis_on: float = 0.55    # score to ENTER a new binding constraint
    hysteresis_off: float = 0.35   # score to EXIT current binding constraint
    confidence_floor: float = 0.30  # minimum confidence to declare a binding constraint


# ---------------------------------------------------------------------------
# Scorer results
# ---------------------------------------------------------------------------

@dataclass
class ScorerResult:
    """Output of one per-family scorer.

    score: None if required signals were absent; otherwise [0, 1].
    evidence: mapping of signal_name → observed_value (for rationale).
    missing: list of signal names that were absent but relevant.
    signal_count: number of signals that were present and used.
    signal_total: total relevant signals for this family.
    """
    score: Optional[float]
    evidence: dict[str, object] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)
    signal_count: int = 0
    signal_total: int = 0

    @property
    def signal_completeness(self) -> float:
        """Fraction of relevant signals that were present."""
        if self.signal_total == 0:
            return 0.0
        return self.signal_count / self.signal_total


# ---------------------------------------------------------------------------
# Link quality lookup (HEURISTIC ordering)
# ---------------------------------------------------------------------------

_TOPOLOGY_LINK_QUALITY: dict[str, float] = {
    "nvswitch": 1.0,
    "nv4": 0.90,
    "nv3": 0.80,
    "nv2": 0.70,
    "nv1": 0.60,
    "pix": 0.50,
    "pxb": 0.40,
    "phb": 0.30,
    "node": 0.25,
    "sys": 0.20,
    "rack": 0.10,
    "region": 0.05,
}


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def _linear_score(val: float, lo: float, hi: float) -> float:
    """Map val in [lo, hi] → [0, 1], clamped."""
    if hi <= lo:
        return 0.0
    return _clamp((val - lo) / (hi - lo))


# ---------------------------------------------------------------------------
# Per-family scorers
# ---------------------------------------------------------------------------

def _score_energy(region: RegionState, cfg: ConstraintConfig) -> ScorerResult:
    """Score energy pressure for a single region.

    Signals used:
    - EnergyState.price_per_mwh       (primary)
    - EnergyState.price_percentile    (secondary enrichment, not required)
    - RegionState.spare_capacity_pct  (context only)

    Scoring: price alone is sufficient to declare a score. Percentile enriches
    confidence but is not required. Both signals are available in isolation.
    """
    energy = region.energy
    if energy is None:
        return ScorerResult(score=None, missing=["energy_state"], signal_total=2)

    missing: list[str] = []
    evidence: dict[str, object] = {}
    scored: list[tuple[float, float]] = []  # (score, weight)
    signal_count = 0
    signal_total = 2

    # Primary: absolute price
    if energy.price_per_mwh is not None:
        price_score = _linear_score(
            energy.price_per_mwh,
            cfg.energy_high_price_mwh,
            cfg.energy_very_high_price_mwh,
        )
        scored.append((price_score, 1.0))  # HEURISTIC: sole primary signal
        evidence["price_per_mwh"] = energy.price_per_mwh
        signal_count += 1
    else:
        missing.append("energy.price_per_mwh")

    # Secondary: percentile adjusts the final score (enrichment only)
    if energy.price_percentile is not None:
        pct_score = _linear_score(energy.price_percentile, cfg.energy_high_percentile, 100.0)
        scored.append((pct_score, 0.40))  # HEURISTIC: secondary weight
        evidence["price_percentile"] = energy.price_percentile
        signal_count += 1
    else:
        missing.append("energy.price_percentile")

    # Context: spare capacity (evidence only)
    if region.spare_capacity_pct is not None:
        evidence["spare_capacity_pct"] = region.spare_capacity_pct
    else:
        missing.append("spare_capacity_pct")

    # Price (primary) is required; percentile only enriches an existing score
    if "energy.price_per_mwh" in missing:
        return ScorerResult(
            score=None,
            evidence=evidence,
            missing=missing,
            signal_count=signal_count,
            signal_total=signal_total,
        )

    if not scored:
        return ScorerResult(
            score=None,
            evidence=evidence,
            missing=missing,
            signal_count=signal_count,
            signal_total=signal_total,
        )

    # Weighted average of available signals
    total_weight = sum(w for _, w in scored)
    raw = sum(s * w for s, w in scored) / total_weight
    return ScorerResult(
        score=_clamp(raw),
        evidence=evidence,
        missing=missing,
        signal_count=signal_count,
        signal_total=signal_total,
    )


def _score_thermal(region: RegionState, cfg: ConstraintConfig) -> ScorerResult:
    """Score thermal pressure for a single region.

    Signals used:
    - ThermalState.throttling_fraction  (primary)
    - ThermalState.max_gpu_temp_c       (primary)
    - GPUState.temp_c (per GPU)         (secondary)
    - GPUState.clocks_event_reasons     (secondary — throttle bitmask)
    """
    missing: list[str] = []
    evidence: dict[str, object] = {}
    score_components: list[tuple[float, float]] = []  # (value, weight)
    signal_count = 0
    signal_total = 4

    thermal = region.thermal

    if thermal is not None:
        if thermal.throttling_fraction is not None:
            throttle_score = _linear_score(
                thermal.throttling_fraction,
                cfg.thermal_throttle_fraction_warn,
                cfg.thermal_throttle_fraction_high,
            )
            score_components.append((throttle_score, 0.50))  # HEURISTIC
            evidence["throttling_fraction"] = thermal.throttling_fraction
            signal_count += 1
        else:
            missing.append("thermal.throttling_fraction")

        if thermal.max_gpu_temp_c is not None:
            temp_score = _linear_score(
                thermal.max_gpu_temp_c,
                cfg.thermal_warn_temp_c,
                cfg.thermal_critical_temp_c,
            )
            score_components.append((temp_score, 0.30))  # HEURISTIC
            evidence["max_gpu_temp_c"] = thermal.max_gpu_temp_c
            signal_count += 1
        else:
            missing.append("thermal.max_gpu_temp_c")
    else:
        missing.extend(["thermal.throttling_fraction", "thermal.max_gpu_temp_c"])

    # Per-GPU secondary signals
    all_temps = [
        gpu.temp_c
        for node in region.nodes.values()
        for gpu in node.gpus.values()
        if gpu.temp_c is not None
    ]
    throttled_gpus = sum(
        1
        for node in region.nodes.values()
        for gpu in node.gpus.values()
        if gpu.clocks_event_reasons is not None and gpu.clocks_event_reasons & 0x8000 != 0
    )
    total_gpus = sum(len(node.gpus) for node in region.nodes.values())

    if all_temps:
        max_temp = max(all_temps)
        temp_score = _linear_score(
            max_temp,
            cfg.thermal_warn_temp_c,
            cfg.thermal_critical_temp_c,
        )
        score_components.append((temp_score, 0.15))  # HEURISTIC
        evidence["max_gpu_temp_from_dcgm"] = max_temp
        signal_count += 1
    else:
        missing.append("gpu.temp_c")

    if total_gpus > 0 and any(
        gpu.clocks_event_reasons is not None
        for node in region.nodes.values()
        for gpu in node.gpus.values()
    ):
        throttle_frac = throttled_gpus / total_gpus
        throttle_score = _linear_score(
            throttle_frac,
            cfg.thermal_throttle_fraction_warn,
            cfg.thermal_throttle_fraction_high,
        )
        score_components.append((throttle_score, 0.05))  # HEURISTIC
        evidence["dcgm_throttled_gpu_fraction"] = throttle_frac
        signal_count += 1
    else:
        missing.append("gpu.clocks_event_reasons")

    if not score_components:
        return ScorerResult(
            score=None,
            evidence=evidence,
            missing=missing,
            signal_count=signal_count,
            signal_total=signal_total,
        )

    total_weight = sum(w for _, w in score_components)
    if total_weight == 0.0:
        return ScorerResult(score=0.0, evidence=evidence, missing=missing,
                            signal_count=signal_count, signal_total=signal_total)

    raw = sum(v * w for v, w in score_components) / total_weight
    return ScorerResult(
        score=_clamp(raw),
        evidence=evidence,
        missing=missing,
        signal_count=signal_count,
        signal_total=signal_total,
    )


def _score_queue(region: RegionState, cfg: ConstraintConfig) -> ScorerResult:
    """Score queue pressure for a single region.

    Signals used:
    - InferenceServiceState.requests_waiting  (primary, per service)
    - InferenceServiceState.queue_time_p99_ms (primary, per service)
    """
    missing: list[str] = []
    evidence: dict[str, object] = {}
    score_components: list[tuple[float, float]] = []
    signal_count = 0
    signal_total = 2

    total_waiting: Optional[float] = None
    max_queue_wait_ms: Optional[float] = None

    for svc_id, svc in region.services.items():
        if svc.requests_waiting is not None:
            total_waiting = (total_waiting or 0.0) + svc.requests_waiting
        # Use p99 if present; fall back to p95 (engine typically only exposes p95)
        wait_ms = svc.queue_time_p99_ms if svc.queue_time_p99_ms is not None else svc.queue_time_p95_ms
        if wait_ms is not None:
            if max_queue_wait_ms is None or wait_ms > max_queue_wait_ms:
                max_queue_wait_ms = wait_ms

    if total_waiting is not None:
        depth_score = _linear_score(total_waiting, cfg.queue_depth_warn, cfg.queue_depth_high)
        score_components.append((depth_score, 0.45))  # HEURISTIC
        evidence["total_requests_waiting"] = total_waiting
        signal_count += 1
    else:
        missing.append("service.requests_waiting")

    if max_queue_wait_ms is not None:
        wait_score = _linear_score(
            max_queue_wait_ms, cfg.queue_wait_warn_ms, cfg.queue_wait_high_ms
        )
        score_components.append((wait_score, 0.55))  # HEURISTIC
        evidence["max_queue_wait_ms"] = max_queue_wait_ms
        signal_count += 1
    else:
        missing.append("service.queue_time_p95_ms")

    if not score_components:
        return ScorerResult(
            score=None,
            evidence=evidence,
            missing=missing,
            signal_count=signal_count,
            signal_total=signal_total,
        )

    total_weight = sum(w for _, w in score_components)
    raw = sum(v * w for v, w in score_components) / total_weight
    return ScorerResult(
        score=_clamp(raw),
        evidence=evidence,
        missing=missing,
        signal_count=signal_count,
        signal_total=signal_total,
    )


def _score_latency(region: RegionState, cfg: ConstraintConfig) -> ScorerResult:
    """Score latency pressure for a single region.

    Signals used:
    - InferenceServiceState.p99_latency_ms  (primary)
    - InferenceServiceState.ttft_p99_ms     (primary)
    - InferenceServiceState.error_rate_pct  (secondary)
    """
    missing: list[str] = []
    evidence: dict[str, object] = {}
    score_components: list[tuple[float, float]] = []
    signal_count = 0
    signal_total = 3

    max_p99_ms: Optional[float] = None
    max_ttft_p99_ms: Optional[float] = None
    max_error_rate_pct: Optional[float] = None

    for svc in region.services.values():
        if svc.p99_latency_ms is not None:
            if max_p99_ms is None or svc.p99_latency_ms > max_p99_ms:
                max_p99_ms = svc.p99_latency_ms
        if svc.ttft_p99_ms is not None:
            if max_ttft_p99_ms is None or svc.ttft_p99_ms > max_ttft_p99_ms:
                max_ttft_p99_ms = svc.ttft_p99_ms
        if svc.error_rate_pct is not None:
            if max_error_rate_pct is None or svc.error_rate_pct > max_error_rate_pct:
                max_error_rate_pct = svc.error_rate_pct

    if max_p99_ms is not None:
        lat_score = _linear_score(
            max_p99_ms, cfg.latency_p99_warn_ms, cfg.latency_p99_high_ms
        )
        score_components.append((lat_score, 0.50))  # HEURISTIC
        evidence["max_p99_latency_ms"] = max_p99_ms
        signal_count += 1
    else:
        missing.append("service.p99_latency_ms")

    if max_ttft_p99_ms is not None:
        ttft_score = _linear_score(
            max_ttft_p99_ms, cfg.ttft_p99_warn_ms, cfg.ttft_p99_high_ms
        )
        score_components.append((ttft_score, 0.40))  # HEURISTIC
        evidence["max_ttft_p99_ms"] = max_ttft_p99_ms
        signal_count += 1
    else:
        missing.append("service.ttft_p99_ms")

    if max_error_rate_pct is not None:
        err_score = _linear_score(
            max_error_rate_pct, cfg.error_rate_warn_pct, cfg.error_rate_high_pct
        )
        score_components.append((err_score, 0.10))  # HEURISTIC
        evidence["max_error_rate_pct"] = max_error_rate_pct
        signal_count += 1
    else:
        missing.append("service.error_rate_pct")

    if not score_components:
        return ScorerResult(
            score=None,
            evidence=evidence,
            missing=missing,
            signal_count=signal_count,
            signal_total=signal_total,
        )

    total_weight = sum(w for _, w in score_components)
    raw = sum(v * w for v, w in score_components) / total_weight
    return ScorerResult(
        score=_clamp(raw),
        evidence=evidence,
        missing=missing,
        signal_count=signal_count,
        signal_total=signal_total,
    )


def _score_communication(region: RegionState, cfg: ConstraintConfig) -> ScorerResult:
    """Score communication-bottleneck pressure for a single region.

    Design: HIGH NVLink traffic alone is NORMAL for NVSwitch workloads and does
    NOT indicate a constraint. The COMMUNICATION constraint fires when GPU
    communication becomes a bottleneck:

    1. NVLink/PCIe traffic is HIGH AND SM utilization is LOW
       → GPUs are waiting for communication, not computing.
    2. PCIe inter-node traffic is high without co-located NVLink
       → suggests cross-node workloads that would benefit from co-location.

    Signals used:
    - GPUState.nvlink_tx/rx_bytes_per_s + sm_active_ratio or util_pct (primary)
    - GPUState.pcie_tx/rx_bytes_per_s                                  (secondary)
    """
    missing: list[str] = []
    evidence: dict[str, object] = {}
    scored: list[tuple[float, float]] = []
    signal_count = 0
    signal_total = 2

    total_nvlink_gbps = 0.0
    total_pcie_gbps = 0.0
    sm_ratios: list[float] = []
    has_nvlink = False
    has_pcie = False

    for node in region.nodes.values():
        for gpu in node.gpus.values():
            if gpu.nvlink_tx_bytes_per_s is not None and gpu.nvlink_rx_bytes_per_s is not None:
                total_nvlink_gbps += (gpu.nvlink_tx_bytes_per_s + gpu.nvlink_rx_bytes_per_s) / 1e9
                has_nvlink = True
            if gpu.pcie_tx_bytes_per_s is not None and gpu.pcie_rx_bytes_per_s is not None:
                total_pcie_gbps += (gpu.pcie_tx_bytes_per_s + gpu.pcie_rx_bytes_per_s) / 1e9
                has_pcie = True
            # Use sm_active_ratio; fall back to util_pct as proxy
            if gpu.sm_active_ratio is not None:
                sm_ratios.append(gpu.sm_active_ratio)
            elif gpu.util_pct is not None:
                sm_ratios.append(gpu.util_pct / 100.0)

    # Primary: communication-bound = high NVLink traffic + low SM activity
    if has_nvlink and sm_ratios:
        avg_sm = sum(sm_ratios) / len(sm_ratios)
        bw_score = _linear_score(
            total_nvlink_gbps, cfg.nvlink_high_gbps, cfg.nvlink_saturated_gbps
        )
        # SM idle score: 1.0 when SM idle, 0.0 when fully active
        sm_idle = 1.0 - avg_sm
        # Communication bound when BOTH bandwidth AND idleness are high (multiplicative)
        comm_score = bw_score * sm_idle  # HEURISTIC
        scored.append((comm_score, 0.70))
        evidence["total_nvlink_gbps"] = total_nvlink_gbps
        evidence["avg_sm_active_ratio"] = avg_sm
        signal_count += 1
    elif has_nvlink:
        # NVLink present but no SM/util data — can't determine if bottlenecked
        missing.append("gpu.sm_active_ratio")
        evidence["total_nvlink_gbps"] = total_nvlink_gbps
    else:
        missing.append("gpu.nvlink_tx/rx_bytes_per_s")

    # Secondary: high PCIe suggests cross-node communication
    if has_pcie:
        pcie_score = _linear_score(total_pcie_gbps, cfg.pcie_high_gbps, cfg.pcie_high_gbps * 3)
        scored.append((pcie_score, 0.30))  # HEURISTIC
        evidence["total_pcie_gbps"] = total_pcie_gbps
        signal_count += 1
    else:
        missing.append("gpu.pcie_tx/rx_bytes_per_s")

    if not scored:
        return ScorerResult(
            score=None,
            evidence=evidence,
            missing=missing,
            signal_count=signal_count,
            signal_total=signal_total,
        )

    total_weight = sum(w for _, w in scored)
    raw = sum(v * w for v, w in scored) / total_weight
    return ScorerResult(
        score=_clamp(raw),
        evidence=evidence,
        missing=missing,
        signal_count=signal_count,
        signal_total=signal_total,
    )


def _score_memory(region: RegionState, cfg: ConstraintConfig) -> ScorerResult:
    """Score memory/cache pressure for a single region (indirect only).

    Does NOT observe KV cache internals. Uses:
    - InferenceServiceState.kv_cache_usage      (0–1 proxy; primary)
    - InferenceServiceState.prefix_cache_hit_rate (0–1; primary)
    - InferenceServiceState.preemptions_total    (rising trend; secondary)
    - GPUState.mem_used_mb / mem_total_mb        (HBM pressure; secondary)
    """
    missing: list[str] = []
    evidence: dict[str, object] = {}
    score_components: list[tuple[float, float]] = []
    signal_count = 0
    signal_total = 4

    max_kv_cache: Optional[float] = None
    min_prefix_hit: Optional[float] = None
    max_preemptions: Optional[float] = None

    for svc in region.services.values():
        if svc.kv_cache_usage is not None:
            if max_kv_cache is None or svc.kv_cache_usage > max_kv_cache:
                max_kv_cache = svc.kv_cache_usage
        if svc.prefix_cache_hit_rate is not None:
            if min_prefix_hit is None or svc.prefix_cache_hit_rate < min_prefix_hit:
                min_prefix_hit = svc.prefix_cache_hit_rate
        if svc.preemptions_total is not None:
            if max_preemptions is None or svc.preemptions_total > max_preemptions:
                max_preemptions = svc.preemptions_total

    if max_kv_cache is not None:
        kv_score = _linear_score(
            max_kv_cache, cfg.kv_cache_warn_fraction, cfg.kv_cache_high_fraction
        )
        score_components.append((kv_score, 0.45))  # HEURISTIC
        evidence["max_kv_cache_usage"] = max_kv_cache
        signal_count += 1
    else:
        missing.append("service.kv_cache_usage")

    if min_prefix_hit is not None:
        # Low hit rate → memory pressure (cold routing or cache eviction)
        low_hit_score = _linear_score(
            1.0 - min_prefix_hit, 1.0 - cfg.prefix_hit_low_fraction, 1.0
        )
        score_components.append((low_hit_score, 0.25))  # HEURISTIC
        evidence["min_prefix_cache_hit_rate"] = min_prefix_hit
        signal_count += 1
    else:
        missing.append("service.prefix_cache_hit_rate")

    # HBM pressure from DCGM
    hbm_fractions = [
        gpu.mem_used_mb / gpu.mem_total_mb
        for node in region.nodes.values()
        for gpu in node.gpus.values()
        if gpu.mem_used_mb is not None and gpu.mem_total_mb is not None and gpu.mem_total_mb > 0
    ]
    if hbm_fractions:
        max_hbm = max(hbm_fractions)
        hbm_score = _linear_score(max_hbm, cfg.hbm_warn_fraction, cfg.hbm_high_fraction)
        score_components.append((hbm_score, 0.20))  # HEURISTIC
        evidence["max_hbm_usage_fraction"] = max_hbm
        signal_count += 1
    else:
        missing.append("gpu.mem_used_mb/mem_total_mb")

    if max_preemptions is not None:
        # Any preemptions suggest KV cache pressure; non-zero is meaningful
        # Soft signal: score 0.3 if >0, scale up to 1.0 at 100 preemptions
        preemption_score = min(1.0, 0.3 + max_preemptions / 100.0) if max_preemptions > 0 else 0.0
        score_components.append((preemption_score, 0.10))  # HEURISTIC
        evidence["max_preemptions_total"] = max_preemptions
        signal_count += 1
    else:
        missing.append("service.preemptions_total")

    if not score_components:
        return ScorerResult(
            score=None,
            evidence=evidence,
            missing=missing,
            signal_count=signal_count,
            signal_total=signal_total,
        )

    total_weight = sum(w for _, w in score_components)
    raw = sum(v * w for v, w in score_components) / total_weight
    return ScorerResult(
        score=_clamp(raw),
        evidence=evidence,
        missing=missing,
        signal_count=signal_count,
        signal_total=signal_total,
    )


def _score_topology(region: RegionState, cfg: ConstraintConfig) -> ScorerResult:
    """Score topology pressure for a single region.

    Signals used:
    - TopologyState.pair_levels (GPU interconnect quality)

    Rationale: if the region has topology data, compute the fraction of GPU
    pairs with weak interconnects. If no topology data, score is None.
    """
    missing: list[str] = []
    evidence: dict[str, object] = {}
    signal_count = 0
    signal_total = 1

    if region.topology is None:
        return ScorerResult(
            score=None,
            missing=["topology_state"],
            signal_count=0,
            signal_total=signal_total,
        )

    topo = region.topology
    pairs = topo.pair_levels
    if not pairs:
        return ScorerResult(
            score=None,
            evidence={"note": "topology present but no GPU pairs"},
            missing=["topology.pair_levels"],
            signal_count=0,
            signal_total=signal_total,
        )

    total_pairs = len(pairs)
    weak_pairs = sum(
        1 for link in pairs.values()
        if _TOPOLOGY_LINK_QUALITY.get(link.value, 0.0) < cfg.topology_weak_link_threshold
    )
    weak_fraction = weak_pairs / total_pairs
    evidence["weak_link_fraction"] = weak_fraction
    evidence["total_gpu_pairs"] = total_pairs
    evidence["interconnect_class"] = topo.interconnect_class
    signal_count += 1

    return ScorerResult(
        score=_clamp(weak_fraction),
        evidence=evidence,
        missing=missing,
        signal_count=signal_count,
        signal_total=signal_total,
    )


def _score_utilization(region: RegionState, cfg: ConstraintConfig) -> ScorerResult:
    """Score underutilization/fragmentation pressure for a single region.

    Signals used:
    - GPUState.util_pct (per GPU)
    """
    missing: list[str] = []
    evidence: dict[str, object] = {}
    signal_count = 0
    signal_total = 1

    util_vals = [
        gpu.util_pct
        for node in region.nodes.values()
        for gpu in node.gpus.values()
        if gpu.util_pct is not None
    ]

    if not util_vals:
        missing.append("gpu.util_pct")
        return ScorerResult(
            score=None,
            evidence=evidence,
            missing=missing,
            signal_count=signal_count,
            signal_total=signal_total,
        )

    avg_util = sum(util_vals) / len(util_vals)
    idle_count = sum(1 for u in util_vals if u < cfg.util_idle_pct)
    idle_fraction = idle_count / len(util_vals)

    # High score = low utilization (pressure to bin-pack)
    avg_score = _linear_score(
        cfg.util_low_cluster_pct - avg_util, 0.0, cfg.util_low_cluster_pct
    )
    idle_score = _linear_score(idle_fraction, 0.0, cfg.util_idle_gpu_fraction)
    raw = avg_score * 0.50 + idle_score * 0.50  # HEURISTIC

    evidence["avg_gpu_util_pct"] = avg_util
    evidence["idle_gpu_fraction"] = idle_fraction
    evidence["gpu_count"] = len(util_vals)
    signal_count += 1

    return ScorerResult(
        score=_clamp(raw),
        evidence=evidence,
        missing=missing,
        signal_count=signal_count,
        signal_total=signal_total,
    )


# ---------------------------------------------------------------------------
# Region-level assessment
# ---------------------------------------------------------------------------

def _assess_region(
    region: RegionState,
    cfg: ConstraintConfig,
    previous_binding: Optional[ConstraintType],
) -> tuple[dict[ConstraintType, float], list[str], dict[str, object], float]:
    """Compute per-family scores for one region.

    Returns (scores, missing_signals, all_evidence, confidence).
    """
    scorer_fns = {
        ConstraintType.ENERGY: _score_energy,
        ConstraintType.THERMAL: _score_thermal,
        ConstraintType.QUEUE: _score_queue,
        ConstraintType.LATENCY: _score_latency,
        ConstraintType.COMMUNICATION: _score_communication,
        ConstraintType.MEMORY: _score_memory,
        ConstraintType.TOPOLOGY: _score_topology,
        ConstraintType.UTILIZATION: _score_utilization,
    }

    scores: dict[ConstraintType, float] = {}
    all_missing: list[str] = []
    all_evidence: dict[str, object] = {}
    completeness_vals: list[float] = []

    for ct, fn in scorer_fns.items():
        result = fn(region, cfg)
        if result.score is not None:
            scores[ct] = result.score
        all_missing.extend(result.missing)
        all_evidence.update({f"{ct.value}.{k}": v for k, v in result.evidence.items()})
        completeness_vals.append(result.signal_completeness)

    # Confidence = mean signal completeness across all families
    confidence = (sum(completeness_vals) / len(completeness_vals)) if completeness_vals else 0.0
    # If is_partial or many missing sources, confidence is additionally penalised upstream
    return scores, all_missing, all_evidence, _clamp(confidence)


# ---------------------------------------------------------------------------
# Constraint classifier
# ---------------------------------------------------------------------------

class ConstraintClassifier:
    """Stateful constraint classifier with hysteresis.

    Usage::

        classifier = ConstraintClassifier()
        assessment = classifier.assess(cluster_state)

    State:
    - ``_previous_binding``: last binding constraint per region (for hysteresis)
    - ``_cfg``: heuristic threshold configuration

    Thread safety: not thread-safe. Use one instance per scheduling loop.
    """

    def __init__(self, cfg: Optional[ConstraintConfig] = None) -> None:
        self._cfg = cfg or ConstraintConfig()
        self._previous_binding: dict[Optional[str], Optional[ConstraintType]] = {}

    def reset(self) -> None:
        """Reset hysteresis state (e.g. after config change)."""
        self._previous_binding.clear()

    def assess(self, state: ClusterState) -> ConstraintAssessment:
        """Classify the binding constraint from a cluster-level ClusterState.

        Produces a cluster-level assessment by aggregating per-region scores.
        The binding constraint is the highest-scoring constraint across all
        regions, subject to hysteresis.

        Missing telemetry reduces confidence and excludes families from scoring.
        It never causes a fabricated binding constraint.
        """
        ts = state.timestamp
        provenance = Provenance(
            source="constraint_classifier",
            fetched_at=ts,
            confidence="high",
            is_sandbox=state.provenance.is_sandbox,
        )

        if not state.regions:
            return ConstraintAssessment(
                timestamp=ts,
                provenance=Provenance(
                    source="constraint_classifier",
                    fetched_at=ts,
                    confidence="low",
                    is_sandbox=state.provenance.is_sandbox,
                ),
                region=None,
                scores={},
                binding_constraint=None,
                confidence=0.0,
                missing_signals=["no_regions"],
                rationale="No regions in cluster state; cannot classify.",
                safe_action_types=["no_op"],
                disallowed_action_types=[],
            )

        # Aggregate per-region scores
        agg_scores: dict[ConstraintType, list[float]] = {}
        all_missing: list[str] = []
        all_evidence: dict[str, object] = {}
        all_confidence: list[float] = []

        for region_id, region in state.regions.items():
            prev = self._previous_binding.get(region_id)
            scores, missing, evidence, conf = _assess_region(region, self._cfg, prev)
            for ct, score in scores.items():
                agg_scores.setdefault(ct, []).append(score)
            all_missing.extend(missing)
            all_evidence.update(evidence)
            all_confidence.append(conf)

        # Aggregate: max score per constraint family across regions
        final_scores: dict[ConstraintType, float] = {
            ct: max(vals) for ct, vals in agg_scores.items()
        }

        confidence = (sum(all_confidence) / len(all_confidence)) if all_confidence else 0.0
        # Penalise confidence if state is partial
        if state.is_partial:
            confidence *= 0.70  # HEURISTIC

        # Determine binding constraint with hysteresis
        previous = self._previous_binding.get(None)  # cluster-level
        binding = self._apply_hysteresis(final_scores, previous, confidence)

        # Build rationale
        rationale = self._build_rationale(
            binding, final_scores, all_missing, all_evidence, confidence
        )

        # Update hysteresis state
        self._previous_binding[None] = binding

        return ConstraintAssessment(
            timestamp=ts,
            provenance=provenance,
            region=None,
            scores=final_scores,
            binding_constraint=binding,
            confidence=_clamp(confidence),
            missing_signals=sorted(set(all_missing)),
            rationale=rationale,
            safe_action_types=list(_SAFE_ACTIONS.get(binding or ConstraintType.NONE, ["no_op"])),
            disallowed_action_types=list(
                _DISALLOWED_ACTIONS.get(binding or ConstraintType.NONE, [])
            ),
        )

    def assess_region(
        self, state: ClusterState, region_id: str
    ) -> ConstraintAssessment:
        """Classify the binding constraint for a specific region.

        Useful when regions have very different workloads/constraints.
        """
        ts = state.timestamp
        if region_id not in state.regions:
            return ConstraintAssessment(
                timestamp=ts,
                provenance=Provenance(
                    source="constraint_classifier",
                    fetched_at=ts,
                    confidence="low",
                    is_sandbox=state.provenance.is_sandbox,
                ),
                region=region_id,
                scores={},
                binding_constraint=None,
                confidence=0.0,
                missing_signals=[f"region_{region_id}_not_found"],
                rationale=f"Region {region_id!r} not found in cluster state.",
                safe_action_types=["no_op"],
                disallowed_action_types=[],
            )

        region = state.regions[region_id]
        prev = self._previous_binding.get(region_id)
        scores, missing, evidence, confidence = _assess_region(region, self._cfg, prev)

        if state.is_partial:
            confidence *= 0.70  # HEURISTIC

        binding = self._apply_hysteresis(scores, prev, confidence)
        self._previous_binding[region_id] = binding

        rationale = self._build_rationale(binding, scores, missing, evidence, confidence)

        provenance = Provenance(
            source="constraint_classifier",
            fetched_at=ts,
            confidence="high" if confidence > 0.6 else ("medium" if confidence > 0.3 else "low"),
            is_sandbox=state.provenance.is_sandbox,
        )

        return ConstraintAssessment(
            timestamp=ts,
            provenance=provenance,
            region=region_id,
            scores=scores,
            binding_constraint=binding,
            confidence=_clamp(confidence),
            missing_signals=sorted(set(missing)),
            rationale=rationale,
            safe_action_types=list(_SAFE_ACTIONS.get(binding or ConstraintType.NONE, ["no_op"])),
            disallowed_action_types=list(
                _DISALLOWED_ACTIONS.get(binding or ConstraintType.NONE, [])
            ),
        )

    def _apply_hysteresis(
        self,
        scores: dict[ConstraintType, float],
        previous_binding: Optional[ConstraintType],
        confidence: float,
    ) -> Optional[ConstraintType]:
        """Apply hysteresis to prevent flapping between binding constraints.

        Rules (HEURISTIC thresholds from ConstraintConfig):
        - To ENTER a new binding state: score >= hysteresis_on AND confidence >= confidence_floor.
        - To EXIT the current binding state: current score falls below hysteresis_off.
        - If no constraint exceeds threshold: return None.
        """
        cfg = self._cfg

        if confidence < cfg.confidence_floor:
            # Too little data to declare anything
            return None

        if not scores:
            return None

        best_ct = max(scores, key=lambda ct: scores[ct])
        best_score = scores[best_ct]

        # Hysteresis OFF: can current binding constraint maintain its position?
        if previous_binding is not None and previous_binding in scores:
            prev_score = scores[previous_binding]
            if prev_score >= cfg.hysteresis_off:
                # Still above the exit threshold — check whether to switch
                if best_ct != previous_binding and best_score > prev_score + 0.10:  # HEURISTIC
                    # New leader is clearly ahead — switch
                    if best_score >= cfg.hysteresis_on:
                        return best_ct
                return previous_binding  # sticky
            # Fell below exit threshold — must re-earn

        # Hysteresis ON: new binding constraint must cross the entry threshold
        if best_score >= cfg.hysteresis_on:
            return best_ct

        return None

    def _build_rationale(
        self,
        binding: Optional[ConstraintType],
        scores: dict[ConstraintType, float],
        missing: list[str],
        evidence: dict[str, object],
        confidence: float,
    ) -> str:
        parts: list[str] = []

        if binding is None:
            parts.append("No binding constraint detected")
        else:
            score = scores.get(binding, 0.0)
            parts.append(f"Binding constraint: {binding.value} (score={score:.2f})")

        if scores:
            top = sorted(scores.items(), key=lambda x: -x[1])[:3]
            top_str = ", ".join(f"{ct.value}={s:.2f}" for ct, s in top)
            parts.append(f"Top scores: {top_str}")

        parts.append(f"Confidence: {confidence:.2f}")

        if missing:
            n = len(set(missing))
            parts.append(f"Missing signals: {n}")

        return "; ".join(parts)
