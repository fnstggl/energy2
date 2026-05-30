"""Convert a normalized trace into Aurelius simulator arrival ticks.

The Aurelius ``ClusterSimulator`` drives arrivals synthetically (diurnal +
Markov-modulated bursts) with a *constant* per-request token proxy
(``_TOKENS_PER_REQUEST`` / ``avg_output_tokens`` in ``engine.py``). To replay a
**real** trace we instead bin the normalized requests into fixed-duration
arrival ticks that preserve, per tick:

  * arrival timestamps (the tick window + measured RPS),
  * real prompt / output / total tokens (means + sums, not a constant),
  * model mix,
  * session / cache-affinity reuse (proxy for prefix locality),
  * log-type mix and failure counts.

These ``ArrivalTick`` objects are the trace-derived "simulator arrivals" that
the BurstGPT backtest feeds through the **unchanged** serving physics in
``aurelius/simulation/cluster/serving.py``. Pure / deterministic, stdlib only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

from .schema import NormalizedLLMRequest


@dataclass(frozen=True)
class ArrivalTick:
    """Aggregated real arrivals for one fixed-duration replay tick."""

    tick_index: int
    start_s: float
    end_s: float
    duration_s: float
    request_count: int
    arrival_rate_rps: float
    prompt_tokens_mean: float
    output_tokens_mean: float
    total_prompt_tokens: int
    total_output_tokens: int
    failures: int
    distinct_cache_keys: int
    # Fraction of this tick's requests whose cache_affinity_key was already seen
    # earlier in the trace — an honest proxy for warm prefix/session locality.
    reuse_fraction: float
    model_mix: dict = field(default_factory=dict)
    log_type_mix: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "tick_index": self.tick_index,
            "start_s": self.start_s,
            "end_s": self.end_s,
            "duration_s": self.duration_s,
            "request_count": self.request_count,
            "arrival_rate_rps": round(self.arrival_rate_rps, 6),
            "prompt_tokens_mean": round(self.prompt_tokens_mean, 4),
            "output_tokens_mean": round(self.output_tokens_mean, 4),
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_output_tokens": self.total_output_tokens,
            "failures": self.failures,
            "distinct_cache_keys": self.distinct_cache_keys,
            "reuse_fraction": round(self.reuse_fraction, 6),
            "model_mix": self.model_mix,
            "log_type_mix": self.log_type_mix,
        }


def requests_to_arrival_ticks(
    requests: Sequence[NormalizedLLMRequest],
    *,
    tick_seconds: float = 60.0,
    count_failures_as_arrivals: bool = True,
) -> list[ArrivalTick]:
    """Bin normalized requests into fixed ``tick_seconds`` arrival ticks.

    Empty interior ticks (no arrivals) are still emitted with zero RPS so the
    replay sees the trace's real idle gaps. Cache-affinity reuse is computed
    against all earlier requests in time order (global warm-key tracking).
    """
    if not requests:
        return []
    if tick_seconds <= 0:
        raise ValueError("tick_seconds must be > 0")

    ordered = sorted(requests, key=lambda r: (r.timestamp_s, r.request_id))
    t0 = ordered[0].timestamp_s
    t_end = ordered[-1].timestamp_s
    n_ticks = max(1, int(math.floor((t_end - t0) / tick_seconds)) + 1)

    buckets: list[list[NormalizedLLMRequest]] = [[] for _ in range(n_ticks)]
    for r in ordered:
        idx = min(n_ticks - 1, int((r.timestamp_s - t0) / tick_seconds))
        buckets[idx].append(r)

    seen_keys: set = set()
    ticks: list[ArrivalTick] = []
    for i, bucket in enumerate(buckets):
        start = t0 + i * tick_seconds
        end = start + tick_seconds
        if not bucket:
            ticks.append(
                ArrivalTick(
                    tick_index=i, start_s=start, end_s=end, duration_s=tick_seconds,
                    request_count=0, arrival_rate_rps=0.0, prompt_tokens_mean=0.0,
                    output_tokens_mean=0.0, total_prompt_tokens=0,
                    total_output_tokens=0, failures=0, distinct_cache_keys=0,
                    reuse_fraction=0.0, model_mix={}, log_type_mix={},
                )
            )
            continue

        count = len(bucket)
        served = [r for r in bucket if not r.is_failure] or bucket
        prompt_sum = sum(r.prompt_tokens for r in bucket)
        output_sum = sum(r.output_tokens for r in bucket)
        prompt_mean = sum(r.prompt_tokens for r in served) / len(served)
        output_mean = sum(r.output_tokens for r in served) / len(served)
        failures = sum(1 for r in bucket if r.is_failure)

        # Reuse is only counted for requests that carry an affinity key. A trace
        # with no session/prefix/logical-stream signal (cache_affinity_key=None,
        # e.g. Azure LLM) gets ZERO reuse — no invented cache benefit.
        reused = 0
        keys_this_tick: set = set()
        for r in bucket:
            key = r.cache_affinity_key
            if key is None:
                continue
            keys_this_tick.add(key)
            if key in seen_keys:
                reused += 1
            seen_keys.add(key)

        model_mix: dict = {}
        log_mix: dict = {}
        for r in bucket:
            model_mix[r.model] = model_mix.get(r.model, 0) + 1
            log_mix[r.log_type] = log_mix.get(r.log_type, 0) + 1

        arrivals = count if count_failures_as_arrivals else (count - failures)
        ticks.append(
            ArrivalTick(
                tick_index=i,
                start_s=start,
                end_s=end,
                duration_s=tick_seconds,
                request_count=count,
                arrival_rate_rps=arrivals / tick_seconds,
                prompt_tokens_mean=prompt_mean,
                output_tokens_mean=output_mean,
                total_prompt_tokens=prompt_sum,
                total_output_tokens=output_sum,
                failures=failures,
                distinct_cache_keys=len(keys_this_tick),
                reuse_fraction=reused / count,
                model_mix=dict(sorted(model_mix.items())),
                log_type_mix=dict(sorted(log_mix.items())),
            )
        )
    return ticks
