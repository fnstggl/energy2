"""Topology / communication / placement realism for the cluster simulator.

Pure, deterministic functions (all randomness is caller-supplied via a
``random.Random`` → seedable) that replace the simulator's simplistic single
link-type topology penalty with a believable communication model: a
latency-bandwidth message cost T(m) = alpha + m/B_eff across DISTINCT fabric
regimes (NVSwitch → NVLink → PCIe → same-rack IB → cross-rack → cross-region),
ring/tree collective amplification, small-message latency amplification, fabric
contention/congestion, synchronization-stall penalties, communication-induced
tail latency, topology telemetry confidence, and a topology-aware migration veto.

Every magnitude comes from ``calibration.TOPOLOGY_PARAMS`` / ``FABRIC_REGIMES`` /
``NVLINK_GENERATIONS`` / ``WORKLOAD_COMM_PROFILES`` (inspectable provenance +
confidence) and is overridable via a per-run ``config`` dict. These are proxies,
NOT a network simulation:

- the fabric regime bandwidths/latencies are vendor-doc-anchored PRIORS, not
  measured per-cluster numbers;
- the topology distance ladder (same-GPU 0 … cross-region 6) is a tunable
  ordinal heuristic, NOT measured hop counts;
- the collective cost equations are the standard ring/tree APPROXIMATIONS, not a
  fitted NCCL model.

Do NOT read any value here as production-accurate. The goal is that
topology-aware placement materially matters: bad placement can collapse
throughput, synchronization-heavy jobs become hard to migrate, communication
penalties can outweigh energy savings, and topology degradation amplifies
p95/p99 faster than the mean.
"""

from __future__ import annotations

import math
import random
from typing import Optional

from .calibration import (
    FABRIC_REGIMES,
    TOPOLOGY_DISTANCE_LADDER,
    FabricRegime,
    WorkloadCommProfile,
    resolve_fabric_regime,
    topology_value,
)

__all__ = [
    "CommRegime",
    "fabric_regime_for_distance",
    "topology_distance",
    "small_message_amplification",
    "effective_bandwidth",
    "congestion_amplifier",
    "message_time_ms",
    "ring_allreduce_ms",
    "tree_collective_ms",
    "collective_latency_ms",
    "collective_amplification",
    "moe_hotspot_amplification",
    "placement_quality_score",
    "communication_penalty",
    "comm_throughput_factor",
    "synchronization_penalty",
    "comm_tail_multipliers",
    "nic_saturation",
    "topology_risk",
    "topology_telemetry_confidence",
    "telemetry_discounted_score",
    "topology_migration_blocked",
]


class CommRegime:
    NOMINAL = "nominal"
    ELEVATED = "elevated"
    CONGESTED = "congested"
    COLLAPSE = "collapse"


_GB = 1e9  # bytes per GB (decimal, matches GB/s convention)


# ---------------------------------------------------------------------------
# Fabric regime + distance ladder
# ---------------------------------------------------------------------------

def topology_distance(regime_name: str) -> int:
    """Topology-distance ladder rung for a regime (same-GPU 0 … cross-region 6)."""
    if regime_name in TOPOLOGY_DISTANCE_LADDER:
        return TOPOLOGY_DISTANCE_LADDER[regime_name]
    r = FABRIC_REGIMES.get(regime_name)
    return r.distance if r is not None else 6


def fabric_regime_for_distance(distance: int) -> FabricRegime:
    """Return the fabric regime at a given distance rung (worst case at rung)."""
    best: Optional[FabricRegime] = None
    for r in FABRIC_REGIMES.values():
        if r.distance == distance:
            # Prefer the slower regime at a shared rung (e.g. node over socket).
            if best is None or r.b_eff_gbps < best.b_eff_gbps:
                best = r
    if best is not None:
        return best
    # Fall back to the nearest higher rung.
    higher = sorted(
        (r for r in FABRIC_REGIMES.values() if r.distance >= distance),
        key=lambda r: r.distance,
    )
    return higher[0] if higher else FABRIC_REGIMES["cross_region"]


# ---------------------------------------------------------------------------
# Latency-bandwidth message cost: T(m) = alpha + m / B_eff
# ---------------------------------------------------------------------------

def small_message_amplification(m_bytes: float, config: Optional[dict] = None) -> float:
    """Latency amplification for small messages (alpha + protocol dominate).

    1.0 for large messages; ramps to small_message_amp_max as m → 0. Small
    messages do NOT enjoy peak bandwidth — latency-bound collectives pay this.
    """
    thresh = topology_value("small_message_bytes", config)
    amp_max = topology_value("small_message_amp_max", config)
    if m_bytes >= thresh or thresh <= 0:
        return 1.0
    frac = 1.0 - max(0.0, m_bytes) / thresh   # 0 at threshold → 1 at zero bytes
    return 1.0 + (amp_max - 1.0) * frac


def congestion_amplifier(load: float, config: Optional[dict] = None) -> float:
    """Latency amplification under congestion: (1/(1-load))^k past the onset.

    1.0 below the onset; rises convexly toward saturation. Clamped so a fully
    saturated link does not produce infinite latency.
    """
    onset = topology_value("congestion_onset", config)
    k = topology_value("congestion_convexity", config)
    load = max(0.0, min(0.999, load))
    if load <= onset:
        return 1.0
    over = (load - onset) / max(1e-6, 1.0 - onset)
    return float((1.0 / max(1e-3, 1.0 - over)) ** k)


def effective_bandwidth(
    regime: FabricRegime, load: float, config: Optional[dict] = None
) -> float:
    """Effective bandwidth (GB/s) of a regime under a given link load [0,1].

    Full bandwidth below the congestion onset; degrades convexly toward a floor
    as the link saturates. The regime's congestion_sensitivity scales how fast
    it collapses (oversubscribed fabrics fall faster).
    """
    onset = topology_value("congestion_onset", config)
    floor = topology_value("congestion_bw_floor", config)
    load = max(0.0, min(0.999, load))
    if load <= onset:
        return regime.b_eff_gbps
    over = (load - onset) / max(1e-6, 1.0 - onset)
    over = min(1.0, over * regime.congestion_sensitivity)
    k = topology_value("congestion_convexity", config)
    factor = (1.0 - over) ** k * (1.0 - floor) + floor
    return regime.b_eff_gbps * max(floor, factor)


def message_time_ms(
    m_bytes: float,
    regime: FabricRegime,
    load: float = 0.0,
    config: Optional[dict] = None,
) -> float:
    """Point-to-point message time in ms: T = (alpha + m/B_eff), with small-message
    amplification on the latency term and congestion degradation of B_eff.
    """
    b_eff = effective_bandwidth(regime, load, config)
    alpha_us = regime.latency_us * small_message_amplification(m_bytes, config)
    alpha_us *= congestion_amplifier(load, config)
    alpha_s = alpha_us / 1e6
    bw_s = m_bytes / max(1.0, b_eff * _GB)
    return (alpha_s + bw_s) * 1000.0


# ---------------------------------------------------------------------------
# Collective communication (ring all-reduce / tree)
# ---------------------------------------------------------------------------

def ring_allreduce_ms(
    m_bytes: float,
    n: int,
    regime: FabricRegime,
    load: float = 0.0,
    config: Optional[dict] = None,
) -> float:
    """Ring all-reduce approximation:

    T_ring = 2(N-1)/N * (m / B_eff) + w * 2(N-1) * alpha

    The bandwidth term saturates near 2x a single send; the latency term grows
    linearly in N (latency-bound for small m / many ranks).
    """
    if n <= 1:
        return 0.0
    b_eff = effective_bandwidth(regime, load, config)
    alpha_us = regime.latency_us * small_message_amplification(m_bytes, config)
    alpha_us *= congestion_amplifier(load, config)
    w = topology_value("allreduce_alpha_term_weight", config)
    bw_s = 2.0 * (n - 1) / n * (m_bytes / max(1.0, b_eff * _GB))
    lat_s = w * 2.0 * (n - 1) * (alpha_us / 1e6)
    return (bw_s + lat_s) * 1000.0


def tree_collective_ms(
    m_bytes: float,
    n: int,
    regime: FabricRegime,
    load: float = 0.0,
    config: Optional[dict] = None,
) -> float:
    """Tree collective approximation: T_tree = log2(N) * alpha + m / B_eff.

    Latency grows logarithmically in N (good for small latency-bound messages).
    """
    if n <= 1:
        return 0.0
    b_eff = effective_bandwidth(regime, load, config)
    alpha_us = regime.latency_us * small_message_amplification(m_bytes, config)
    alpha_us *= congestion_amplifier(load, config)
    lat_s = math.log2(n) * (alpha_us / 1e6)
    bw_s = m_bytes / max(1.0, b_eff * _GB)
    return (lat_s + bw_s) * 1000.0


def moe_hotspot_amplification(load: float, config: Optional[dict] = None) -> float:
    """All-to-all hotspot amplification for MoE traffic under congestion.

    1.0 with no congestion; rises toward moe_hotspot_amp as expert-popularity
    skew creates incast hotspots.
    """
    amp = topology_value("moe_hotspot_amp", config)
    cong = max(0.0, min(1.0, load))
    return 1.0 + (amp - 1.0) * cong


def collective_latency_ms(
    collective: str,
    m_bytes: float,
    n: int,
    regime: FabricRegime,
    load: float,
    rng: random.Random,
    config: Optional[dict] = None,
) -> float:
    """Modelled collective latency (ms) for a collective kind, with jitter.

    Dispatches to ring (all_reduce), tree (tree), or a single message (p2p);
    all_to_all uses ring scaled by the MoE hotspot amplification. Multiplicative
    jitter (routing variation / NIC contention) means topology behaviour is NOT
    one deterministic curve.
    """
    coll = (collective or "none").lower()
    if coll == "none" or n <= 1:
        base = message_time_ms(m_bytes, regime, load, config) if n > 1 else 0.0
    elif coll == "all_reduce":
        base = ring_allreduce_ms(m_bytes, n, regime, load, config)
    elif coll == "tree":
        base = tree_collective_ms(m_bytes, n, regime, load, config)
    elif coll == "all_to_all":
        base = ring_allreduce_ms(m_bytes, n, regime, load, config) * moe_hotspot_amplification(
            load, config
        )
    else:  # p2p
        base = message_time_ms(m_bytes, regime, load, config)
    jitter = topology_value("collective_jitter_frac", config)
    return max(0.0, base * (1.0 + rng.gauss(0.0, jitter)))


def collective_amplification(
    collective: str,
    m_bytes: float,
    n: int,
    regime: FabricRegime,
    load: float = 0.0,
    config: Optional[dict] = None,
) -> float:
    """Collective cost relative to a single point-to-point send of the same m.

    >1 means the collective amplifies communication cost (clamped to
    collective_amp_max). Used to make all-reduce / all-to-all sensitivity show
    up in throughput and tails.
    """
    if n <= 1:
        return 1.0
    p2p = message_time_ms(m_bytes, regime, load, config)
    if p2p <= 0:
        return 1.0
    coll = (collective or "none").lower()
    if coll == "all_reduce":
        t = ring_allreduce_ms(m_bytes, n, regime, load, config)
    elif coll == "tree":
        t = tree_collective_ms(m_bytes, n, regime, load, config)
    elif coll == "all_to_all":
        t = ring_allreduce_ms(m_bytes, n, regime, load, config) * moe_hotspot_amplification(
            load, config
        )
    else:
        return 1.0
    amp_max = topology_value("collective_amp_max", config)
    return max(1.0, min(amp_max, t / p2p))


# ---------------------------------------------------------------------------
# Placement quality score (topology distance ladder → 0-1 quality)
# ---------------------------------------------------------------------------

def _regime_quality(regime: FabricRegime) -> float:
    """0-1 placement quality of a single bottleneck regime.

    Derived from the calibrated regime priors: a bandwidth ratio (vs the best
    inter-GPU regime, NVSwitch) blended with a latency ratio. NVSwitch ≈ 1.0;
    cross-region ≈ 0. The 0.65/0.35 blend is an engineering weighting (bandwidth
    matters more for collectives), NOT a measured coefficient.
    """
    best = FABRIC_REGIMES["nvswitch"]
    bw_ratio = min(1.0, regime.b_eff_gbps / best.b_eff_gbps) ** 0.5
    lat_ratio = min(1.0, best.latency_us / max(regime.latency_us, best.latency_us))
    return max(0.0, min(1.0, 0.65 * bw_ratio + 0.35 * lat_ratio))


def placement_quality_score(
    regime_names: list[str],
    traffic_weights: Optional[list[float]] = None,
    config: Optional[dict] = None,
) -> tuple[float, float]:
    """Topology quality for a placement from its pairwise regimes.

    Implements S = Σ(w_ij * d(i,j)) as a traffic-weighted mean distance, returned
    alongside a 0-1 quality score (1 = ideal locality). The quality is driven by
    the WORST (bottleneck) regime — a collective is paced by its slowest hop —
    blended with the traffic-weighted mean so a single bad hop dominates but
    mostly-local placements still score well.

    Returns (distance_score, quality_score). distance_score is the raw Σ w*d
    (lower = better); quality_score is in [0,1] (higher = better).
    """
    if not regime_names:
        return 0.0, 1.0
    weights = traffic_weights or [1.0] * len(regime_names)
    total_w = sum(weights) or 1.0
    distances = [topology_distance(rn) for rn in regime_names]
    distance_score = sum(w * d for w, d in zip(weights, distances)) / total_w

    regimes = [resolve_fabric_regime(rn) for rn in regime_names]
    worst = max(regimes, key=lambda r: r.distance)
    q_worst = _regime_quality(worst)
    q_mean = sum(w * _regime_quality(r) for w, r in zip(weights, regimes)) / total_w
    quality = 0.7 * q_worst + 0.3 * q_mean
    return distance_score, max(0.0, min(1.0, quality))


# ---------------------------------------------------------------------------
# Communication penalty + throughput / synchronization / tail effects
# ---------------------------------------------------------------------------

def communication_penalty(
    profile: WorkloadCommProfile,
    m_bytes: float,
    n: int,
    regime: FabricRegime,
    quality_score: float,
    load: float = 0.0,
    config: Optional[dict] = None,
) -> float:
    """Communication penalty P = λ * (m/B_eff + alpha) / (μ * S).

    A normalized, comparable scalar (informational + used in migration/placement
    decisions): higher = communication is more costly relative to the placement
    quality S. Scales with the workload's communication weight λ and shrinks as
    placement quality improves.
    """
    mu = topology_value("topology_penalty_mu", config)
    s = max(0.05, quality_score)
    b_eff = effective_bandwidth(regime, load, config)
    alpha_s = regime.latency_us / 1e6
    per_msg_s = m_bytes / max(1.0, b_eff * _GB) + alpha_s
    coll_amp = collective_amplification(
        profile.collective, m_bytes, n, regime, load, config
    )
    return profile.comm_weight * per_msg_s * coll_amp / (mu * s)


def comm_throughput_factor(
    profile: WorkloadCommProfile,
    quality_score: float,
    load: float = 0.0,
    config: Optional[dict] = None,
) -> float:
    """Throughput multiplier in [1 - max, 1] from communication overhead.

    Slowdown grows with poor topology (1 - quality), the workload's
    communication weight λ, and congestion. A tensor-parallel job (λ=1) split
    off NVSwitch can collapse to the floor; a comm-light job is barely touched.
    """
    smax = topology_value("topology_throughput_penalty_max", config)
    deficit = 1.0 - max(0.0, min(1.0, quality_score))
    cong = max(0.0, min(1.0, load))
    drive = profile.comm_weight * (deficit + 0.5 * cong * profile.comm_weight)
    slowdown = smax * min(1.0, drive)
    return max(1.0 - smax, 1.0 - slowdown)


def synchronization_penalty(
    profile: WorkloadCommProfile,
    quality_score: float,
    load: float,
    rng: random.Random,
    config: Optional[dict] = None,
) -> tuple[float, float]:
    """Synchronization-stall penalty for a bulk-synchronous workload.

    Returns (straggler_frac, slowdown_frac). The slowest rank sets the pace, so
    the penalty grows with topology deficit, congestion, and per-rank jitter.
    Non-sync-heavy workloads return (0, 0).
    """
    if not profile.sync_heavy:
        return 0.0, 0.0
    deficit = 1.0 - max(0.0, min(1.0, quality_score))
    cong = max(0.0, min(1.0, load))
    jitter = topology_value("sync_straggler_jitter", config)
    straggler = max(0.0, deficit * 0.6 + cong * 0.4 + abs(rng.gauss(0.0, jitter)))
    straggler = min(1.0, straggler)
    smax = topology_value("sync_penalty_max", config)
    slowdown = smax * straggler
    return straggler, max(0.0, min(smax, slowdown))


def comm_tail_multipliers(
    profile: WorkloadCommProfile,
    quality_score: float,
    load: float = 0.0,
    config: Optional[dict] = None,
) -> tuple[float, float]:
    """Communication-induced tail multipliers (p95, p99) on top of queueing tails.

    At good topology / low congestion these sit near the base; as topology
    degrades or the fabric congests they grow convexly toward comm_tail_max — and
    p99 grows faster than p95 (tails blow up super-linearly). Scaled by the
    workload's tail sensitivity.
    """
    p95_base = topology_value("comm_tail_p95_base", config)
    p99_base = topology_value("comm_tail_p99_base", config)
    tmax = topology_value("comm_tail_max", config)
    deficit = 1.0 - max(0.0, min(1.0, quality_score))
    cong = max(0.0, min(1.0, load))
    # Combined degradation, weighted by the workload's tail sensitivity.
    deg = min(1.0, (0.6 * deficit + 0.4 * cong) * (0.4 + profile.tail_sensitivity))
    # Multiplicative amplification: p99 fans out faster than p95 (different
    # exponents), so the tail does NOT converge to one shared ceiling and p99
    # genuinely blows up faster than p95 (and far faster than the mean).
    amp = 1.0 + deg * (tmax - 1.0)
    p95 = max(1.0, p95_base * amp ** 0.85)
    p99 = max(p95, p99_base * amp ** 1.2)
    return p95, p99


def nic_saturation(
    cross_node_traffic_frac: float, nic_throughput_frac: float,
    config: Optional[dict] = None,
) -> tuple[float, bool]:
    """NIC saturation + incast flag for cross-node communication.

    Returns (saturation, incast_active). Cross-node collectives bottleneck on the
    NIC before the intra-node fabric; past the onset, incast amplifies latency.
    """
    onset = topology_value("nic_congestion_onset", config)
    sat = max(0.0, min(1.0, cross_node_traffic_frac * max(0.0, nic_throughput_frac)))
    return sat, sat > onset


def topology_risk(
    profile: WorkloadCommProfile,
    quality_score: float,
    load: float = 0.0,
    config: Optional[dict] = None,
) -> tuple[float, bool]:
    """Topology degradation risk [0,1] + collective-instability flag.

    Risk rises with topology deficit, congestion, and communication weight.
    Instability triggers for sync-heavy / TP-class workloads when the quality
    score falls below the instability threshold.
    """
    deficit = 1.0 - max(0.0, min(1.0, quality_score))
    cong = max(0.0, min(1.0, load))
    risk = min(1.0, profile.comm_weight * (0.6 * deficit + 0.4 * cong))
    unstable = (
        (profile.sync_heavy or profile.comm_weight >= 0.7)
        and quality_score < topology_value("tp_instability_score", config)
    )
    return risk, unstable


# ---------------------------------------------------------------------------
# Telemetry confidence + topology-aware migration governor
# ---------------------------------------------------------------------------

def topology_telemetry_confidence(
    nvlink_visible: bool,
    pcie_visible: bool,
    nic_visible: bool,
    stale_ticks: int,
    detached_devices: int = 0,
) -> str:
    """Map topology telemetry visibility/staleness to a confidence tier.

    HIGH   full NVLink + PCIe + NIC visibility, fresh, no detached devices.
    MEDIUM one map missing or mildly stale.
    LOW    multiple missing / very stale / detached devices. Missing topology
    telemetry LOWERS confidence — it must NOT be read as ideal proximity.
    """
    visible = sum([bool(nvlink_visible), bool(pcie_visible), bool(nic_visible)])
    if visible == 3 and stale_ticks <= 1 and detached_devices == 0:
        return "high"
    if visible >= 2 and stale_ticks <= 3 and detached_devices == 0:
        return "medium"
    return "low"


def telemetry_discounted_score(
    quality_score: float, telemetry_tier: str, config: Optional[dict] = None
) -> float:
    """Discount an optimistic topology score under poor telemetry confidence.

    With LOW confidence we assume the placement MIGHT be worse than reported, so
    the usable score is capped (missing topology ≠ ideal proximity). MEDIUM
    applies a milder cap. HIGH passes through.
    """
    if telemetry_tier == "high":
        return quality_score
    cap = topology_value("topology_confidence_min_score", config)
    if telemetry_tier == "low":
        return min(quality_score, cap)
    # medium: halfway between the cap and the reported score
    return min(quality_score, 0.5 * (quality_score + cap))


def topology_migration_blocked(
    profile: WorkloadCommProfile,
    dest_distance: int,
    dest_quality_score: float,
    telemetry_tier: str,
    config: Optional[dict] = None,
) -> bool:
    """Veto migrating a communication-sensitive workload across fabric domains.

    Blocks when the workload is communication-heavy enough (λ ≥ threshold) AND
    the destination breaks fabric locality (distance ≥ veto distance) OR the
    destination placement would be unstable for a sync-heavy / TP-class job. With
    LOW telemetry confidence the effective distance threshold is lowered (missing
    topology ≠ safe → be conservative).
    """
    comm_thresh = topology_value("migration_veto_comm_weight", config)
    if profile.comm_weight < comm_thresh:
        return False
    veto_dist = topology_value("migration_veto_distance", config)
    if telemetry_tier == "low":
        veto_dist -= 1.0  # be more conservative when topology is unclear
    if dest_distance >= veto_dist:
        return True
    if profile.sync_heavy and dest_quality_score < topology_value(
        "tp_instability_score", config
    ):
        return True
    return False
