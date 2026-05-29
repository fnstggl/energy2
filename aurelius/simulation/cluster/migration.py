"""Migration / rerouting / drain / cold-start realism for the simulator.

Pure, deterministic functions (all randomness is caller-supplied via a
``random.Random`` → seedable) that price the *operational* cost of moving or
rerouting a workload. They make migration expensive and risky so that naive
energy-arbitrage / aggressive rerouting can LOSE, warm replicas become valuable,
and phased rollouts beat abrupt ones.

Every magnitude comes from ``calibration.MIGRATION_PARAMS`` /
``calibration.ENGINE_STARTUP_PROFILES`` (inspectable provenance + confidence)
and is overridable via a per-run ``config`` dict. These are proxies, not a
control-plane simulation:

- drain/cold-start times are documented operational anchors (e.g. the K8s 30s
  grace period) shaped into heavy-tailed distributions, NOT measured per-cluster
  numbers;
- the migration cost is a sum of believable terms, NOT a fitted model;
- the governor / rollout logic encodes operational heuristics, not a controller.

Do NOT read any value here as production-accurate. See the realism-gap report.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Optional

from .calibration import (
    EngineStartupProfile,
    migration_value,
    resolve_engine_profile,
)

__all__ = [
    "ColdStartBreakdown",
    "MigrationCost",
    "sample_lognormal_scaled",
    "drain_seconds",
    "pdb_blocks_migration",
    "reroute_seconds",
    "proxy_saturation_factor",
    "cache_loss_penalty_ms",
    "cold_start_seconds",
    "scaleup_seconds",
    "seconds_to_warmup_ticks",
    "batch_efficiency_under_churn",
    "tail_uplift",
    "migration_cost",
    "migration_veto_reason",
    "next_traffic_fraction",
    "should_rollback",
    "resolve_engine_profile",
]


# ---------------------------------------------------------------------------
# Deterministic distribution helpers (rng supplied by caller)
# ---------------------------------------------------------------------------

def sample_lognormal_scaled(rng: random.Random, mean: float, sigma: float) -> float:
    """Right-skewed sample whose *median* is `mean` (lognormal), bounded ≥ 0.

    Using median=mean keeps the central tendency at the documented anchor while
    producing a heavy upper tail (slow drains / cold starts). sigma controls the
    tail weight. Deterministic given rng.
    """
    if mean <= 0:
        return 0.0
    mu = math.log(mean)
    return max(0.0, rng.lognormvariate(mu, max(0.0, sigma)))


# ---------------------------------------------------------------------------
# Kubernetes drain
# ---------------------------------------------------------------------------

def drain_seconds(rng: random.Random, config: Optional[dict] = None) -> float:
    """T_drain = T_evict + T_grace + T_rebind (seconds), heavy-tailed grace.

    The graceful-termination window is a truncated right-skew around the K8s
    default (30s) — actual shutdown may finish earlier or drag out. Eviction and
    rebind are modelled as their anchors with mild skew.
    """
    t_evict = migration_value("drain_evict_seconds", config)
    grace_mean = migration_value("drain_grace_seconds", config)
    skew = migration_value("drain_grace_skew", config)
    t_rebind = migration_value("drain_rebind_seconds", config)
    # Grace is right-skewed but capped at 2× the configured window (most apps
    # shut down within the grace period; a tail drags out).
    t_grace = min(2.0 * grace_mean, sample_lognormal_scaled(rng, grace_mean, skew))
    return t_evict + t_grace + t_rebind


def pdb_blocks_migration(pdb_available: int) -> bool:
    """A drain/migration is blocked when the PodDisruptionBudget allows 0 evictions."""
    return pdb_available <= 0


# ---------------------------------------------------------------------------
# Request rerouting + proxy bottleneck
# ---------------------------------------------------------------------------

def reroute_seconds(
    proxy_queue_s: float, network_rtt_s: float, replica_accept_s: float
) -> float:
    """T_route = max(proxy_queue, network_rtt, replica_accept).

    The slowest stage dominates — a saturated proxy can make rerouting expensive
    even when the network and replica are fast.
    """
    return max(0.0, proxy_queue_s, network_rtt_s, replica_accept_s)


def proxy_saturation_factor(
    offered_rps: float, replicas: int, config: Optional[dict] = None,
    cap_per_override: Optional[float] = None,
) -> float:
    """Proxy/ingress queue amplification ≥ 1.0 as offered load nears capacity.

    Capacity = proxy_capacity_rps_per_replica × replicas. Replica count alone
    does NOT determine throughput: past ~capacity the proxy queues convexly
    (1/(1-load))^k. This lets proxy bottlenecks dominate queue wait / TTFT.

    ``cap_per_override`` (a per-ingress capacity from the queue) takes precedence
    over the global config default when provided.
    """
    cap_per = (
        cap_per_override if cap_per_override is not None
        else migration_value("proxy_capacity_rps_per_replica", config)
    )
    k = migration_value("proxy_saturation_convexity", config)
    capacity = max(1e-6, cap_per * max(1, replicas))
    load = max(0.0, offered_rps) / capacity
    if load < 0.7:
        return 1.0
    load = min(0.999, load)
    base = (1.0 / (1.0 - load)) ** k
    safe = (1.0 / (1.0 - 0.7)) ** k
    # Cap at a believable maximum: a saturated proxy dominates queue wait without
    # producing absurd telemetry. The downstream queue-wait cap (60s) bounds the
    # actual latency effect regardless.
    return min(100.0, base / safe)


# ---------------------------------------------------------------------------
# Cache-loss penalty (ΔT_prefill) — prompt-length scaled
# ---------------------------------------------------------------------------

def cache_loss_penalty_ms(
    prompt_tokens: float,
    hit_rate_after_move: float,
    prefill_cost_per_token_ms: float,
) -> float:
    """ΔT_prefill = prompt_tokens · prefill_cost_per_token · (1 − hit_after).

    Scales with prompt length and collapses-to-cold (hit_after ≈ 0) on a reroute
    to a replica with no shared state. NOT a fixed penalty.
    """
    miss = 1.0 - max(0.0, min(1.0, hit_rate_after_move))
    return max(0.0, prompt_tokens) * max(0.0, prefill_cost_per_token_ms) * miss


# ---------------------------------------------------------------------------
# Cold start (heavy-tailed, engine-specific, bimodal first-compile)
# ---------------------------------------------------------------------------

@dataclass
class ColdStartBreakdown:
    t_node: float
    t_pull: float
    t_load: float
    t_gpu_transfer: float
    t_warmup: float
    first_compile: bool

    @property
    def total_seconds(self) -> float:
        return self.t_node + self.t_pull + self.t_load + self.t_gpu_transfer + self.t_warmup


def cold_start_seconds(
    engine: str,
    rng: random.Random,
    config: Optional[dict] = None,
    *,
    profile: Optional[EngineStartupProfile] = None,
) -> ColdStartBreakdown:
    """Decomposed, heavy-tailed cold start for a serving engine.

    T_cold = T_node + T_pull + T_load + T_gpu_transfer + T_warmup, each stage a
    right-skewed lognormal around its anchor. Bimodal: with probability
    coldstart_firstcompile_prob the warmup stage hits the first-compile path
    (×coldstart_firstcompile_mult) — and compile-heavy engines (TensorRT-LLM)
    pay a large warmup even on the warm path. NOT a single Gaussian.
    """
    prof = profile if profile is not None else resolve_engine_profile(engine)
    sigma = migration_value("coldstart_lognormal_sigma", config)
    fc_prob = migration_value("coldstart_firstcompile_prob", config)
    fc_mult = migration_value("coldstart_firstcompile_mult", config)

    t_node = sample_lognormal_scaled(rng, prof.t_node, sigma) if prof.t_node > 0 else 0.0
    t_pull = sample_lognormal_scaled(rng, prof.t_pull, sigma)
    t_load = sample_lognormal_scaled(rng, prof.t_load, sigma)
    t_gpu = sample_lognormal_scaled(rng, prof.t_gpu_transfer, sigma)
    t_warmup = sample_lognormal_scaled(rng, prof.t_warmup, sigma)

    first_compile = rng.random() < fc_prob
    if first_compile:
        # compile-heavy engines amplify the first-compile path further
        mult = fc_mult * (1.5 if prof.compile_heavy else 1.0)
        t_warmup *= mult

    return ColdStartBreakdown(t_node, t_pull, t_load, t_gpu, t_warmup, first_compile)


def scaleup_seconds(
    engine: str,
    rng: random.Random,
    config: Optional[dict] = None,
    *,
    from_zero: bool = False,
) -> float:
    """T_scaleup = T_scheduling + T_imagepull + T_modelload + T_warmup (seconds).

    Reuses the cold-start decomposition for pull/load/warmup and adds a
    scheduling delay. Scale-from-zero is not latency-neutral; the amplification
    on TTFT is applied separately via scale_from_zero_ttft_mult.
    """
    sched = migration_value("scaleup_scheduling_seconds", config)
    cs = cold_start_seconds(engine, rng, config)
    return sched + cs.t_pull + cs.t_load + cs.t_gpu_transfer + cs.t_warmup


def seconds_to_warmup_ticks(total_seconds: float, tick_duration_hours: float) -> int:
    """Convert a startup duration to whole warmup ticks (≥ 1 if any startup).

    At coarse (hourly) tick granularity a multi-minute cold start is sub-tick, so
    the floor of 1 ensures at least one degraded warmup tick; the residual TTFT
    cost is injected separately as a startup penalty.
    """
    if total_seconds <= 0:
        return 0
    tick_seconds = max(1.0, tick_duration_hours * 3600.0)
    return max(1, math.ceil(total_seconds / tick_seconds))


# ---------------------------------------------------------------------------
# Batching disruption under churn
# ---------------------------------------------------------------------------

def batch_efficiency_under_churn(
    base_efficiency: float, churn_rate: float, config: Optional[dict] = None
) -> float:
    """η_batch ∈ (0, 1] that degrades with reroute churn.

    churn_rate is a non-negative measure of recent migrations/reroutes. Decode
    cohorts fragment and batch occupancy collapses under churn → throughput and
    latency suffer. Floors at batch_churn_floor.
    """
    floor = migration_value("batch_churn_floor", config)
    sens = migration_value("batch_churn_sensitivity", config)
    decay = math.exp(-sens * max(0.0, churn_rate))
    eff = base_efficiency * (floor + (1.0 - floor) * decay)
    return max(0.01, min(1.0, eff))


# ---------------------------------------------------------------------------
# Migration tail uplift (p95/p99, NOT p50-only)
# ---------------------------------------------------------------------------

def tail_uplift(
    rollout_instability: float,
    queue_pressure: float,
    churn_rate: float,
    cache_loss: float,
    config: Optional[dict] = None,
) -> float:
    """Percentile uplift multiplier (≥ base) for p95/p99 during/after migration.

    Combines four normalized [0,1]-ish drivers into a convex uplift between
    tail_uplift_base and tail_uplift_max. Migration amplifies the TAIL, not just
    the median.
    """
    base = migration_value("tail_uplift_base", config)
    mmax = migration_value("tail_uplift_max", config)
    drivers = (
        max(0.0, min(1.0, rollout_instability))
        + max(0.0, min(1.0, queue_pressure))
        + max(0.0, min(1.0, churn_rate))
        + max(0.0, min(1.0, cache_loss))
    ) / 4.0
    return base + (mmax - base) * (drivers ** 2)


# ---------------------------------------------------------------------------
# Composite migration cost
# ---------------------------------------------------------------------------

@dataclass
class MigrationCost:
    """C_mig decomposed into its operational terms (ms unless noted)."""
    t_transfer_ms: float     # image pull + model load + GPU alloc + transfer
    t_warmup_ms: float       # graph capture / compile / runtime warmup
    t_requeue_ms: float      # drain wait + routing/rebind delay
    t_cacheloss_ms: float    # lost prefix reuse (cold-route prefill)
    t_batchloss_factor: float  # η_batch ∈ (0,1] (throughput degradation factor)
    t_tail_mult: float       # p95/p99 uplift multiplier
    cold_start: ColdStartBreakdown
    drain_s: float

    @property
    def startup_penalty_ms(self) -> float:
        """One-shot TTFT penalty injected over the warmup window."""
        return self.t_transfer_ms + self.t_warmup_ms + self.t_requeue_ms + self.t_cacheloss_ms


def migration_cost(
    engine: str,
    prompt_tokens: float,
    hit_rate_before: float,
    prefill_cost_per_token_ms: float,
    rng: random.Random,
    *,
    base_batch_efficiency: float = 1.0,
    churn_rate: float = 0.0,
    rollout_instability: float = 0.0,
    queue_pressure: float = 0.0,
    network_rtt_ms: Optional[float] = None,
    from_zero: bool = False,
    config: Optional[dict] = None,
) -> MigrationCost:
    """Full C_mig = T_transfer + T_warmup + T_requeue + T_cacheloss + T_batchloss + T_tail.

    Deterministic given rng. The destination replica is cold (hit_after ≈ 0), so
    the cache-loss term is priced at the full pre-move hit rate.
    """
    cs = cold_start_seconds(engine, rng, config)
    t_transfer_ms = (cs.t_node + cs.t_pull + cs.t_load + cs.t_gpu_transfer) * 1000.0
    t_warmup_ms = cs.t_warmup * 1000.0

    drain_s = drain_seconds(rng, config)
    rtt_s = (network_rtt_ms if network_rtt_ms is not None
             else migration_value("reroute_network_rtt_ms", config)) / 1000.0
    accept_s = migration_value("reroute_replica_accept_ms", config) / 1000.0
    route_s = reroute_seconds(0.0, rtt_s, accept_s)
    t_requeue_ms = (drain_s + route_s) * 1000.0

    # Cache loss: rerouted requests hit a cold replica (hit_after ≈ 0).
    t_cacheloss_ms = cache_loss_penalty_ms(prompt_tokens, 0.0, prefill_cost_per_token_ms) \
        * max(0.0, min(1.0, hit_rate_before))

    eta_batch = batch_efficiency_under_churn(base_batch_efficiency, churn_rate, config)
    cache_loss_norm = max(0.0, min(1.0, hit_rate_before))
    tail_mult = tail_uplift(rollout_instability, queue_pressure, churn_rate,
                            cache_loss_norm, config)

    if from_zero:
        tail_mult *= migration_value("scale_from_zero_ttft_mult", config)

    return MigrationCost(
        t_transfer_ms=t_transfer_ms,
        t_warmup_ms=t_warmup_ms,
        t_requeue_ms=t_requeue_ms,
        t_cacheloss_ms=t_cacheloss_ms,
        t_batchloss_factor=eta_batch,
        t_tail_mult=tail_mult,
        cold_start=cs,
        drain_s=drain_s,
    )


# ---------------------------------------------------------------------------
# Migration governor (veto logic)
# ---------------------------------------------------------------------------

def migration_veto_reason(
    *,
    queue_depth: float,
    locality_confidence: float,
    p95_unstable: bool,
    rollout_instability: float,
    pdb_available: int,
    warmup_incomplete: bool,
    startup_heavy: bool,
    scale_from_zero: bool,
    config: Optional[dict] = None,
) -> Optional[str]:
    """Return a veto reason if migration should be blocked, else None.

    Encodes operational restraint: do-nothing is often safest. Block when queue
    pressure is high, cache affinity is strong, p95/rollout are unstable, the PDB
    forbids eviction, the workload is still warming, or a startup-heavy /
    scale-from-zero path would amplify the cost.
    """
    if pdb_blocks_migration(pdb_available):
        return "pdb_unavailable"
    q_thresh = migration_value("governor_queue_pressure_qdepth", config)
    if queue_depth >= q_thresh:
        return "queue_pressure_high"
    if locality_confidence >= 0.7:
        return "cache_affinity_strong"
    if p95_unstable:
        return "p95_unstable"
    if rollout_instability >= 0.5:
        return "rollout_instability_high"
    if warmup_incomplete:
        return "warmup_incomplete"
    if startup_heavy:
        return "startup_heavy_path"
    if scale_from_zero:
        return "scale_from_zero_active"
    return None


# ---------------------------------------------------------------------------
# Phased rollout / traffic shifting
# ---------------------------------------------------------------------------

# Canary-style progressive fractions; advance only when stable.
_ROLLOUT_FRACTIONS = (0.1, 0.25, 0.5, 1.0)


def next_traffic_fraction(current_fraction: float, stable: bool) -> float:
    """Advance the phased-rollout traffic fraction iff the current phase is stable.

    Returns the next fraction in the canary ladder when stable, else holds.
    """
    if not stable:
        return current_fraction
    for f in _ROLLOUT_FRACTIONS:
        if f > current_fraction + 1e-9:
            return f
    return 1.0


def should_rollback(
    p99_latency_ms: float, sla_p99_ms: float, config: Optional[dict] = None
) -> bool:
    """Roll back a rollout phase if p99 blows past the SLA budget multiple."""
    if sla_p99_ms <= 0:
        return False
    mult = migration_value("rollback_p99_budget_mult", config)
    return p99_latency_ms > sla_p99_ms * mult
