"""GPU utilization / fragmentation / bin-packing realism for the simulator.

Pure, deterministic functions (all randomness is caller-supplied via a
``random.Random`` → seedable) that replace the simulator's scalar GPU-utilization
heuristic with a multidimensional model: SM / DRAM-bandwidth / scheduler / PCIe /
KV utilization with a bottleneck ``U_gpu = min(...)``, a roofline-style token
ceiling, continuous-batching gains with diminishing returns, KV/VRAM headroom,
multidimensional + topology-aware fragmentation, stranded capacity, a saturating
consolidation benefit with a nonlinear risk sum, queue amplification under
packing density, GPU-sharing interference, and utilization telemetry confidence.

Every magnitude comes from ``calibration.UTILIZATION_PARAMS`` /
``WORKLOAD_CLASS_PROFILES`` (inspectable provenance + confidence) and is
overridable via a per-run ``config`` dict. These are proxies, NOT a scheduler /
allocator simulation:

- the utilization regimes (inference Triangular(0.40,0.55,0.70), training
  Triangular(0.85,0.90,0.95)) are configurable PRIORS, not universal targets;
- the roofline ceilings and bottleneck onsets are dimensionless operational
  heuristics, not measured FLOP/byte rooflines;
- the fragmentation thresholds and consolidation curves are engineering priors.

Do NOT read any value here as production-accurate. The goal is that "free GPUs"
are often unusable, consolidation has nonlinear risk, aggressive packing can
destabilize workloads, batching benefits flatten, and utilization becomes a
multidimensional systems problem rather than a scalar occupancy metric.
"""

from __future__ import annotations

import math
import random
from typing import Optional

from .calibration import utilization_value

__all__ = [
    "UtilBottleneck",
    "FragRegime",
    "triangular_sample",
    "dram_bandwidth_demand",
    "scheduler_cap",
    "pcie_cap",
    "memory_bandwidth_cap",
    "effective_utilization",
    "util_throughput_factor",
    "roofline_tokens_per_sec",
    "batching_gain",
    "vram_headroom",
    "kv_headroom",
    "fragmentation_score",
    "topology_fragmentation_score",
    "fragmentation_regime",
    "stranded_breakdown",
    "consolidation_benefit",
    "consolidation_risk",
    "packing_unsafe",
    "queue_amplification",
    "underutilized",
    "utilization_paradox",
    "sharing_interference",
    "cross_node_shard_penalty",
    "bin_packing_risk",
    "util_telemetry_confidence",
]


class UtilBottleneck:
    SM = "sm"
    MEM = "mem"
    SCHED = "sched"
    PCIE = "pcie"
    KV = "kv"


class FragRegime:
    NOMINAL = "nominal"
    ELEVATED = "elevated"
    CRITICAL = "critical"


# ---------------------------------------------------------------------------
# Utilization priors
# ---------------------------------------------------------------------------

def triangular_sample(lo: float, mode: float, hi: float, rng: random.Random) -> float:
    """Deterministic triangular sample in [lo, hi] with the given mode."""
    if hi <= lo:
        return lo
    mode = min(max(mode, lo), hi)
    return rng.triangular(lo, hi, mode)


# ---------------------------------------------------------------------------
# Multi-dimensional utilization: U_gpu = min(U_sm, U_mem, U_sched, U_pcie)
# ---------------------------------------------------------------------------

def dram_bandwidth_demand(
    mem_bytes_per_token: float, batch_occupancy: float, config: Optional[dict] = None
) -> float:
    """DRAM-bandwidth demand fraction for a workload (decode is memory-bound).

    Rises with per-token memory traffic and batch occupancy (more concurrent
    sequences → more KV reads). Can exceed 1.0 to indicate a saturated,
    bandwidth-bound regime; callers clamp the telemetry value.
    """
    batch_occupancy = max(0.0, min(1.0, batch_occupancy))
    return max(0.0, mem_bytes_per_token * (0.3 + 0.5 * batch_occupancy))


def memory_bandwidth_cap(dram_demand: float, config: Optional[dict] = None) -> float:
    """Throughput cap (≤1) from the memory-bandwidth dimension.

    1.0 while DRAM demand is below the saturation onset; below 1.0 once the
    workload is bandwidth-bound (decode throughput pinned by DRAM bandwidth).
    """
    onset = utilization_value("mem_bw_saturation_onset", config)
    if dram_demand <= onset:
        return 1.0
    return max(0.05, onset / max(1e-6, dram_demand))


def scheduler_cap(active_sequences: float, config: Optional[dict] = None) -> float:
    """Throughput cap (≤1) from the scheduler/service limit S_sched.

    1.0 below the scheduler capacity; below 1.0 once admission + scheduling
    overhead dominates (too many concurrent sequences).
    """
    cap = utilization_value("scheduler_capacity_seqs", config)
    if active_sequences <= cap or cap <= 0:
        return 1.0
    return max(0.05, cap / max(1e-6, active_sequences))


def pcie_cap(pcie_pressure: float, config: Optional[dict] = None) -> float:
    """Throughput cap (≤1) from PCIe transfer pressure.

    1.0 below the PCIe pressure onset; below 1.0 once host<->device staging
    suppresses effective occupancy.
    """
    onset = utilization_value("pcie_pressure_onset", config)
    pcie_pressure = max(0.0, min(1.0, pcie_pressure))
    if pcie_pressure <= onset:
        return 1.0
    over = (pcie_pressure - onset) / max(1e-6, 1.0 - onset)
    return max(0.2, 1.0 - 0.6 * over)


def effective_utilization(
    sm_util: float, mem_cap: float, sched_cap: float, pcie_cap_val: float,
) -> tuple[float, str]:
    """U_gpu = min(U_sm, U_sm·mem_cap, U_sm·sched_cap, U_sm·pcie_cap).

    Returns (effective_utilization, bottleneck_dimension). The effective
    utilization is the SM utilization scaled by the tightest non-compute cap; the
    bottleneck names which dimension binds. Low SM with a tight cap means the GPU
    is busy-but-throttled (memory/scheduler/PCIe bound), NOT idle.
    """
    sm_util = max(0.0, min(1.0, sm_util))
    dims = {
        UtilBottleneck.SM: 1.0,
        UtilBottleneck.MEM: max(0.0, min(1.0, mem_cap)),
        UtilBottleneck.SCHED: max(0.0, min(1.0, sched_cap)),
        UtilBottleneck.PCIE: max(0.0, min(1.0, pcie_cap_val)),
    }
    bottleneck = min(dims, key=lambda k: dims[k])
    return sm_util * dims[bottleneck], bottleneck


def util_throughput_factor(mem_cap: float, sched_cap: float, pcie_cap_val: float) -> float:
    """Throughput multiplier in (0,1] = the tightest non-compute capacity cap.

    1.0 when compute is the bottleneck (default well-provisioned case → no
    change); below 1.0 when the workload is memory-bandwidth / scheduler / PCIe
    bound. This is what makes the utilization paradox (high resource use, low
    throughput) and scheduler/DRAM bottlenecks materially cut throughput.
    """
    return max(0.05, min(1.0, mem_cap, sched_cap, pcie_cap_val))


# ---------------------------------------------------------------------------
# Roofline token ceiling
# ---------------------------------------------------------------------------

def roofline_tokens_per_sec(
    f_peak: float, f_tok: float, bw_peak: float, b_tok: float,
    s_sched: float, k_kv: float,
) -> tuple[float, str]:
    """tokens/sec ≈ min(F_peak/f_tok, BW_peak/b_tok, S_sched, K_kv).

    Returns (tokens_per_sec, binding_term). Constrains batching gains and creates
    memory-bandwidth / scheduler / KV-capacity ceilings.
    """
    terms = {
        "compute": f_peak / f_tok if f_tok > 0 else math.inf,
        "memory": bw_peak / b_tok if b_tok > 0 else math.inf,
        "scheduler": s_sched,
        "kv": k_kv,
    }
    binding = min(terms, key=lambda k: terms[k])
    return max(0.0, terms[binding]), binding


# ---------------------------------------------------------------------------
# Continuous batching gain (diminishing returns)
# ---------------------------------------------------------------------------

def batching_gain(
    output_len_cv: float,
    concurrency: float,
    kv_pressure: float,
    scheduler_pressure: float,
    config: Optional[dict] = None,
) -> float:
    """Continuous-batching throughput gain = 1 + a·CV(output_len), with diminishing
    returns and flattening under KV / scheduler pressure.

    Capped at the COMMON regime (batching_gain_common_max ~8x); the optimistic
    vendor regime (~23x) is reachable only under highly favorable conditions
    (very high CV + concurrency + no pressure). NOT linear; NOT universal.
    """
    a = utilization_value("batching_gain_cv_coeff", config)
    common_max = utilization_value("batching_gain_common_max", config)
    vendor_max = utilization_value("batching_gain_vendor_max", config)
    cv = max(0.0, output_len_cv)
    conc = max(0.0, min(1.0, concurrency))
    # Base gain from length variance, scaled by how much concurrency is available
    # to exploit it (low concurrency → little batching headroom).
    raw = 1.0 + a * cv * (0.3 + 0.7 * conc)
    # Diminishing returns: concave saturation toward the common cap.
    gain = 1.0 + (common_max - 1.0) * (1.0 - math.exp(-(raw - 1.0)))
    # The optimistic vendor regime only opens up with very high CV + concurrency.
    if cv > 1.0 and conc > 0.9:
        favor = min(1.0, (cv - 1.0)) * (conc - 0.9) / 0.1
        gain += (vendor_max - common_max) * favor
    # Pressure flattens the gain (KV thinning + scheduler stalls).
    pressure = max(0.0, min(1.0, max(kv_pressure, scheduler_pressure)))
    gain = 1.0 + (gain - 1.0) * (1.0 - 0.7 * pressure)
    return max(1.0, min(vendor_max, gain))


# ---------------------------------------------------------------------------
# KV / VRAM headroom
# ---------------------------------------------------------------------------

def vram_headroom(used_frac: float, config: Optional[dict] = None) -> tuple[float, bool]:
    """Return (usable_headroom_frac, over_reserve) for a VRAM occupancy.

    Reserves ~vram_headroom_frac (~5%); occupancy above (1 - reserve) eats into
    the reserve and is flagged. gpu_memory_utilization = 1.0 is NOT safe.
    """
    reserve = utilization_value("vram_headroom_frac", config)
    safe_ceiling = 1.0 - reserve
    headroom = max(0.0, safe_ceiling - max(0.0, used_frac))
    return headroom, used_frac > safe_ceiling


def kv_headroom(occupancy: float, config: Optional[dict] = None) -> tuple[float, bool]:
    """Return (headroom_frac, admission_suppressed) for a KV occupancy.

    Headroom shrinks toward the safe occupancy ceiling; above it, admission is
    suppressed (the scheduler hesitates) and preemption risk rises.
    """
    safe = utilization_value("safe_occupancy_max", config)
    headroom = max(0.0, safe - max(0.0, occupancy))
    return headroom, occupancy > safe


# ---------------------------------------------------------------------------
# Fragmentation + stranded capacity
# ---------------------------------------------------------------------------

def fragmentation_score(free_capacity: int, schedulable_capacity: int) -> float:
    """F = 1 - schedulable_free_capacity / free_capacity (0 = none, 1 = total).

    Free-but-unschedulable capacity (wrong domain / VRAM / topology) is fragmented.
    """
    if free_capacity <= 0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - schedulable_capacity / free_capacity))


def topology_fragmentation_score(
    free_by_domain: dict[str, int], demand_by_domain: dict[str, int]
) -> float:
    """F = 1 - Σ min(free_d, demand_d) / Σ free_d over placement domains.

    Captures split topology domains: free GPUs scattered across domains that
    cannot jointly satisfy a domain-local demand are stranded.
    """
    total_free = sum(max(0, v) for v in free_by_domain.values())
    if total_free <= 0:
        return 0.0
    usable = sum(
        min(max(0, free_by_domain.get(d, 0)), max(0, demand_by_domain.get(d, 0)))
        for d in free_by_domain
    )
    return max(0.0, min(1.0, 1.0 - usable / total_free))


def fragmentation_regime(score: float, config: Optional[dict] = None) -> str:
    """Classify a fragmentation score into nominal / elevated / critical."""
    elevated = utilization_value("fragmentation_elevated", config)
    critical = utilization_value("fragmentation_critical", config)
    if score >= critical:
        return FragRegime.CRITICAL
    if score >= elevated:
        return FragRegime.ELEVATED
    return FragRegime.NOMINAL


def stranded_breakdown(
    topology_isolated: int, vram_isolated: int, comm_isolated: int,
    sla_incompatible: int,
) -> int:
    """Total stranded (free-but-unusable) GPUs across all isolation reasons."""
    return max(0, topology_isolated) + max(0, vram_isolated) + max(
        0, comm_isolated
    ) + max(0, sla_incompatible)


# ---------------------------------------------------------------------------
# Consolidation benefit + risk
# ---------------------------------------------------------------------------

def consolidation_benefit(
    consolidation_fraction: float, config: Optional[dict] = None
) -> float:
    """Saturating consolidation benefit = B_max·(1 - exp(-k·fraction)).

    Returns diminish: the first packed jobs free the most idle capacity.
    """
    b_max = utilization_value("consolidation_benefit_max", config)
    k = utilization_value("consolidation_benefit_k", config)
    frac = max(0.0, min(1.0, consolidation_fraction))
    return b_max * (1.0 - math.exp(-k * frac))


def consolidation_risk(
    cross_domain_traffic: float,
    queue_pressure: float,
    inverse_temp_margin: float,
    kv_pressure: float,
    scheduler_pressure: float,
    config: Optional[dict] = None,
) -> float:
    """Consolidation risk R = r1·cross_domain + r2·queue + r3·inv_temp_margin
    + r4·kv + r5·scheduler (clamped to [0,1]).

    Packing becomes unsafe as cross-node sharding, queue pressure, thermal
    tightness, KV pressure, and scheduler pressure rise.
    """
    r1 = utilization_value("consolidation_risk_cross_domain", config)
    r2 = utilization_value("consolidation_risk_queue", config)
    r3 = utilization_value("consolidation_risk_thermal", config)
    r4 = utilization_value("consolidation_risk_kv", config)
    r5 = utilization_value("consolidation_risk_scheduler", config)

    def _c(x: float) -> float:
        return max(0.0, min(1.0, x))

    risk = (
        r1 * _c(cross_domain_traffic)
        + r2 * _c(queue_pressure)
        + r3 * _c(inverse_temp_margin)
        + r4 * _c(kv_pressure)
        + r5 * _c(scheduler_pressure)
    )
    return max(0.0, min(1.0, risk))


def packing_unsafe(risk: float, config: Optional[dict] = None) -> bool:
    """True if the consolidation risk exceeds the unsafe-packing threshold."""
    return risk >= utilization_value("packing_unsafe_risk", config)


# ---------------------------------------------------------------------------
# Queue amplification + sharing + sharding
# ---------------------------------------------------------------------------

def queue_amplification(
    packing_pressure: float, config: Optional[dict] = None
) -> tuple[float, bool]:
    """Queue waiting-time amplification under aggressive packing.

    ``packing_pressure`` is per-replica oversubscription (batch occupancy / load),
    NOT raw GPU allocation — a fully-allocated-but-not-oversubscribed cluster is
    healthy. 1.0 below the onset; rises convexly (less slack to absorb bursts)
    toward an unstable regime, bounded so it compounds with — rather than swamps —
    the serving layer's own saturation amplifier. Returns (amplification, unstable).
    """
    onset = utilization_value("queue_amp_onset", config)
    k = utilization_value("queue_amp_convexity", config)
    d = max(0.0, min(0.999, packing_pressure))
    if d <= onset:
        return 1.0, False
    over = (d - onset) / max(1e-6, 1.0 - onset)
    amp = min(8.0, (1.0 / max(0.12, 1.0 - over)) ** k)
    return amp, over > 0.7


def sharing_interference(
    tenants: int, mode: str, config: Optional[dict] = None
) -> float:
    """Interference fraction from GPU sharing (MIG / time-slice / fractional).

    Scales with the number of co-located tenants; time-slicing interferes more
    than hardware-partitioned MIG. Sharing is NOT free.
    """
    if tenants <= 1 or (mode or "none") == "none":
        return 0.0
    base = utilization_value("gpu_sharing_interference", config)
    mode_mult = {"mig": 0.6, "fractional": 1.0, "time_slice": 1.4}.get(mode, 1.0)
    return max(0.0, min(0.9, base * mode_mult * (tenants - 1)))


def cross_node_shard_penalty(
    node_count: int, topology_sensitivity: float, config: Optional[dict] = None
) -> float:
    """Throughput penalty fraction for sharding a workload across nodes.

    Rises with the number of nodes and the workload's topology sensitivity.
    Cross-node sharding is NOT cheap for communication-heavy workloads.
    """
    if node_count <= 1:
        return 0.0
    sens = max(0.0, min(1.0, topology_sensitivity))
    spread = min(1.0, (node_count - 1) / 3.0)
    return max(0.0, min(0.8, sens * spread))


def bin_packing_risk(
    fragmentation: float, density: float, demand_gpus: int,
    config: Optional[dict] = None,
) -> tuple[float, bool]:
    """Bin-packing risk for placing a demand of ``demand_gpus`` GPUs.

    Rises with fragmentation and density; larger demands are harder to place.
    Returns (risk, unsafe).
    """
    frag = max(0.0, min(1.0, fragmentation))
    dens = max(0.0, min(1.0, density))
    demand_factor = min(1.0, demand_gpus / 8.0)
    risk = min(1.0, 0.5 * frag + 0.3 * dens + 0.2 * demand_factor)
    return risk, risk >= utilization_value("packing_unsafe_risk", config)


# ---------------------------------------------------------------------------
# Underutilization / paradox / telemetry
# ---------------------------------------------------------------------------

def underutilized(sm_util: float, config: Optional[dict] = None) -> bool:
    """True if SM utilization is below the underutilization threshold.

    A packing CANDIDATE — NOT a guarantee of safe consolidation.
    """
    return sm_util < utilization_value("underutilization_sm_threshold", config)


def utilization_paradox(
    sm_util: float, dram_active: float, config: Optional[dict] = None
) -> bool:
    """True under the utilization paradox: high DRAM_ACTIVE + low SM utilization.

    The GPU is busy (memory-bound) but compute-underutilized — high resource use,
    low throughput. Must NOT be read as a safe packing opportunity.
    """
    sm_low = sm_util < utilization_value("underutilization_sm_threshold", config)
    dram_high = dram_active >= utilization_value("paradox_dram_high", config)
    return sm_low and dram_high


def util_telemetry_confidence(
    gpu_util_visible: bool, dram_visible: bool, scheduler_visible: bool,
    stale_ticks: int,
) -> str:
    """Map utilization-telemetry visibility/staleness to a confidence tier.

    HIGH   full GPU_UTIL + DRAM + scheduler visibility, fresh.
    MEDIUM one missing or mildly stale.
    LOW    multiple missing / very stale. Missing telemetry LOWERS packing
    confidence — it must NOT be read as schedulable.
    """
    visible = sum([bool(gpu_util_visible), bool(dram_visible), bool(scheduler_visible)])
    if visible == 3 and stale_ticks <= 1:
        return "high"
    if visible >= 2 and stale_ticks <= 3:
        return "medium"
    return "low"
