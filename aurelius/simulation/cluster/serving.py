"""Inference-serving realism layer for the cluster simulator.

Pure, deterministic functions that make the simulator's serving DYNAMICS
qualitatively believable. They replace three audited over-simplifications:

1. queue wait was linear-ish and p95 was a fixed 3x of the mean →
   now convex near saturation with tails that explode super-linearly.
2. TTFT/TPOT were fixed multipliers of one base →
   now decomposed into queue + prompt + active-sequence + KV-pressure (TTFT) and
   decode-contention (TPOT).
3. throughput scaled linearly with replicas →
   now a batching tradeoff: spreading the same load over more replicas pushes
   each below the batching knee, lowering throughput per GPU.

Every magnitude comes from ``calibration.SERVING_PARAMS`` (inspectable provenance
+ confidence) and is overridable via a per-run ``config`` dict. These functions
are intentionally proxies, not a serving-engine simulation — see the realism-gap
report. All randomness is caller-supplied (seedable) for determinism.
"""

from __future__ import annotations

import random
from typing import Optional

from .calibration import serving_value


def step_burst_state(in_burst: bool, rng: random.Random, config: Optional[dict] = None) -> bool:
    """Advance a Markov-modulated burst state (bursty, not smooth, arrivals).

    Two-state MMPP: quiet → burst with prob `burst_state_prob`; burst → quiet
    with prob `burst_exit_prob`. Deterministic given the RNG.
    """
    if in_burst:
        return rng.random() > serving_value("burst_exit_prob", config)
    return rng.random() < serving_value("burst_state_prob", config)


def arrival_multiplier(in_burst: bool, config: Optional[dict] = None) -> float:
    """Multiplier applied to the base arrival rate while in a burst state."""
    return serving_value("burst_multiplier", config) if in_burst else 1.0


def erlang_c_wait_s(lam: float, mu: float, c: int) -> float:
    """Erlang-C mean waiting time (seconds) for an M/M/c queue.

    lam: arrival rate (req/s), mu: per-server service rate (req/s), c: servers.
    Returns 0 when underloaded with spare servers; grows as rho→1. Saturation
    amplification is applied separately by `saturation_amplifier`.
    """
    if mu <= 0 or c <= 0:
        return float("inf")
    rho = lam / (c * mu)
    if rho <= 0:
        return 0.0
    if rho >= 1.0:
        return float("inf")
    a = lam / mu  # offered load (Erlangs)
    # Erlang-C probability of waiting (numerically stable via Erlang-B recursion).
    # Erlang-B: B(0)=1; B(k) = a*B(k-1) / (k + a*B(k-1))
    b = 1.0
    for k in range(1, c + 1):
        b = (a * b) / (k + a * b)
    # Erlang-C from Erlang-B: C = B / (1 - rho*(1 - B))
    denom = 1.0 - rho * (1.0 - b)
    pw = b / denom if denom > 0 else 1.0
    pw = max(0.0, min(1.0, pw))
    # Mean wait in queue = Pw / (c*mu - lam)
    return pw / (c * mu - lam)


def saturation_amplifier(rho: float, config: Optional[dict] = None) -> float:
    """Convex amplification of waiting time as utilization rises.

    1.0 within the safe band; grows convexly as (1/(1-rho))^convexity once rho
    exceeds `safe_utilization`; very large in the overload region. This is what
    makes 'do nothing' sometimes optimal and aggressive load-piling dangerous.
    """
    safe = serving_value("safe_utilization", config)
    overload = serving_value("overload_utilization", config)
    convexity = serving_value("saturation_convexity", config)
    rho = max(0.0, min(0.999, rho))
    if rho <= safe:
        return 1.0
    # Convex blow-up beyond the safe band.
    base = (1.0 / (1.0 - rho)) ** convexity
    safe_base = (1.0 / (1.0 - safe)) ** convexity
    amp = base / safe_base
    if rho >= overload:
        # Overload-collapse region: extra runaway factor.
        amp *= 1.0 + 5.0 * (rho - overload) / max(1e-6, 1.0 - overload)
    return amp


def tail_multipliers(rho: float, config: Optional[dict] = None) -> tuple[float, float]:
    """(p95/p50, p99/p50) latency tail ratios that GROW with utilization.

    At low load tails are mild; near saturation they explode super-linearly and
    p99 grows faster than p95 (which grows faster than the mean).
    """
    rho = max(0.0, min(0.999, rho))
    # Smooth, convex ramp from base→max as rho→1.
    ramp = rho ** 2
    p95 = serving_value("tail_p95_base", config) + ramp * (
        serving_value("tail_p95_max", config) - serving_value("tail_p95_base", config)
    )
    p99 = serving_value("tail_p99_base", config) + ramp * (
        serving_value("tail_p99_max", config) - serving_value("tail_p99_base", config)
    )
    return p95, p99


def batching_efficiency(active_seqs: float, replicas: int, config: Optional[dict] = None) -> float:
    """Per-replica batching efficiency in (0, 1].

    Continuous batching is most efficient when each replica runs full batches
    (active sequences per replica ≥ the knee). Spreading the same load over more
    replicas pushes each below the knee → lower throughput per GPU. This is the
    replica/batching tradeoff the audit said was missing: more replicas relieve
    the queue but cost batching efficiency; fewer replicas batch better but
    destabilize the queue.
    """
    if replicas <= 0:
        return 1.0
    knee = serving_value("batch_efficiency_knee", config)
    floor = serving_value("batch_efficiency_floor", config)
    per_replica = active_seqs / replicas
    # Ramp up to 1.0 at the knee; floor reflects that a single stream is not
    # throughput-bound (it still gets ~half of full batch throughput).
    eff = per_replica / knee if knee > 0 else 1.0
    return max(floor, min(1.0, eff))


def ttft_ms(
    t_queue_ms: float,
    prompt_tokens: float,
    active_seqs: float,
    kv_pressure: float,
    warmup_factor: float = 1.0,
    config: Optional[dict] = None,
) -> float:
    """Decomposed time-to-first-token (ms).

    TTFT = queue wait + prefill(prompt) + scheduler contention(active seqs)
           + KV-pressure stall, all divided by warmup (cold replicas are slower).
    """
    alpha = serving_value("ttft_per_prompt_token_ms", config)
    beta = serving_value("ttft_per_active_seq_ms", config)
    gamma = serving_value("ttft_kv_pressure_ms", config)
    kv = max(0.0, min(1.0, kv_pressure))
    prefill = alpha * max(0.0, prompt_tokens)
    contention = beta * max(0.0, active_seqs)
    # KV pressure stalls allocation non-linearly (worse past ~0.7 usage).
    kv_stall = gamma * (max(0.0, kv - 0.7) / 0.3) ** 2 if kv > 0.7 else 0.0
    wf = max(0.1, warmup_factor)
    return (t_queue_ms + (prefill + contention + kv_stall)) / wf


def tpot_ms(
    base_tpot_ms: float,
    active_tokens: float,
    throttle_factor: float = 1.0,
    config: Optional[dict] = None,
) -> float:
    """Decomposed inter-token latency (ms): base × throttle + decode contention.

    More active tokens in the decode batch raise per-token latency (the
    throughput/latency tradeoff of continuous batching).
    """
    per_tok = serving_value("tpot_per_active_token_ms", config)
    return base_tpot_ms * max(1.0, throttle_factor) + per_tok * max(0.0, active_tokens)
