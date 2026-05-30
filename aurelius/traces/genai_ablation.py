"""Ablation / attribution harness for the Alibaba GenAI 2026 serving win.

This is a **measurement-only** module. It does NOT add optimizer logic and does
NOT change any constant — it re-composes the EXISTING mechanisms in
``aurelius/traces/genai_backtest.py`` (the ``affinity`` cold-start flag and the
five sizing strategies already implemented there) into a factorial grid so the
+89.5% ``constraint_aware`` vs ``sla_aware`` gain can be attributed.

Two orthogonal existing knobs are varied:

  * **sizing strategy** — how replicas-per-tick are chosen:
      - ``static_peak``      (the ``fifo`` policy's sizing)
      - ``reactive_sla``     (the ``sla_aware`` policy's sizing)
      - ``queue_target``     (the ``queue_aware`` policy's sizing)
      - ``util_target``      (the ``utilization_aware`` policy's sizing)
      - ``anticipatory_sla`` (the ``constraint_aware`` policy's sizing)
  * **affinity** — the model-affinity / warm-pool cold-start lever
    (``_effective_service_s(..., affinity=True)``). In the implemented optimizer
    **"prewarm" and "model-affinity" are the SAME mechanism** (route to a warm
    replica ⇒ avoid reloading the model); there is no separate prewarm constant
    to ablate, so ``+prewarm`` ≡ ``+affinity`` here — stated honestly, not faked.

Each config calls the existing ``genai_backtest`` primitives verbatim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from aurelius.benchmarks.economics import (
    InfrastructureCostConfig,
    compute_economic_kpi,
)

from . import genai_backtest as gb
from .schema import NormalizedGenAIRequest

# (config_name, sizing_strategy, affinity). Mirrors the existing policies +
# their affinity-toggled counterparts. "BestFit" is a bin-packing baseline from
# the Alibaba-GPU/Philly backtests and is NOT a GenAI serving policy; the
# strongest GenAI serving baseline (the docs/RESULTS.md headline) is sla_aware,
# so the requested "BestFit"/"BestFit + affinity" rows map to sla_aware here.
ABLATION_CONFIGS: tuple[tuple[str, str, bool], ...] = (
    ("fifo", "static_peak", False),
    ("fifo_plus_affinity", "static_peak", True),            # == FIFO + prewarm
    ("sla_aware", "reactive_sla", False),                   # headline / "BestFit" analog
    ("sla_aware_plus_affinity", "reactive_sla", True),      # "BestFit + affinity"
    ("queue_aware", "queue_target", False),
    ("queue_aware_plus_affinity", "queue_target", True),
    ("utilization_aware", "util_target", False),
    ("utilization_aware_plus_affinity", "util_target", True),
    ("constraint_aware", "anticipatory_sla", True),         # full optimizer
    ("constraint_aware_no_affinity", "anticipatory_sla", False),  # == no prewarm
)


@dataclass
class AblationResult:
    name: str
    sizing: str
    affinity: bool
    goodput_per_dollar: Optional[float]
    sla_compliant_requests: int
    completed_requests: int
    infra_cost: float
    replica_hours: float
    e2e_p99_s: float
    timeout_rate_pct: float
    mean_cold_start_s: float

    def summary(self) -> dict:
        return {
            "config": self.name, "sizing": self.sizing, "affinity": self.affinity,
            "sla_safe_goodput_per_infra_dollar": (
                None if self.goodput_per_dollar is None
                else round(self.goodput_per_dollar, 4)),
            "sla_compliant_requests": self.sla_compliant_requests,
            "completed_requests": self.completed_requests,
            "infra_cost": round(self.infra_cost, 2),
            "replica_gpu_hours": round(self.replica_hours, 2),
            "e2e_latency_s_p99": round(self.e2e_p99_s, 2),
            "timeout_rate_pct": round(self.timeout_rate_pct, 3),
            "mean_cold_start_s": round(self.mean_cold_start_s, 2),
        }


def _run_config(name, sizing, affinity, ticks, cold, tick_hours) -> AblationResult:
    """Run one ablation config by composing existing genai_backtest primitives."""
    active = [t for t in ticks if t.n > 0]
    if active:
        peak = max(active, key=lambda t: t.arrival_rate)
        static_replicas = gb._size_for_target(peak, cold, affinity, gb.TARGET_RHO_SLA)
    else:
        static_replicas = gb.MIN_REPLICAS

    tokens, timeouts, gpu_hours_tick = [], [], []
    e2e99, weights = [], []
    replica_hours = 0.0
    cold_sum = 0.0
    cold_w = 0
    scale_events = 0
    prev_r = None
    prev_tick = None
    ewma = 0.0

    for t in ticks:
        if t.n > 0:
            ewma = 0.5 * t.arrival_rate + 0.5 * ewma if ewma else t.arrival_rate
        if sizing == "static_peak":
            r = static_replicas
        elif sizing == "anticipatory_sla":
            if t.n:
                plan = gb.TickAgg(t.tick_index, t.start_s, t.n,
                                  max(t.arrival_rate, ewma), t.mean_exec_s,
                                  t.distinct_models, t.lora_frac, t.controlnet_frac,
                                  t.failures)
                r = gb._size_for_sla(plan, cold, affinity)
            else:
                r = gb.MIN_REPLICAS
        elif sizing == "reactive_sla":
            src = prev_tick if prev_tick is not None else t
            r = gb._size_for_sla(src, cold, affinity) if (src and src.n) else gb.MIN_REPLICAS
        elif sizing == "queue_target":
            src = prev_tick if prev_tick is not None else t
            r = (gb._size_for_target(src, cold, affinity, gb.TARGET_RHO_SLA)
                 if (src and src.n) else gb.MIN_REPLICAS)
        elif sizing == "util_target":
            r = (gb._size_for_target(t, cold, affinity, gb.TARGET_RHO_UTIL)
                 if t.n else gb.MIN_REPLICAS)
        else:  # pragma: no cover
            raise ValueError(sizing)

        ev = gb._eval_tick(t, r, cold, affinity)
        tokens.append(t.n)
        timeouts.append(ev["timeout"])
        gpu_hours = r * tick_hours
        gpu_hours_tick.append({"genai-gpu": gpu_hours})
        replica_hours += gpu_hours
        if t.n > 0:
            e2e99.append(ev["e2e_p99"])
            weights.append(t.n)
            cold_sum += ev["cold_s"] * t.n
            cold_w += t.n
        if prev_r is not None and r != prev_r:
            scale_events += 1
        prev_r = r
        prev_tick = t

    cfg = InfrastructureCostConfig(gpu_hour_prices={"genai-gpu": gb.GPU_HOUR_PRICE},
                                   fallback_gpu_hour_price=gb.GPU_HOUR_PRICE)
    kpi = compute_economic_kpi(
        tokens_per_tick=tokens, timeout_rate_pct_per_tick=timeouts,
        energy_cost_per_tick=[0.0] * len(ticks),
        active_gpu_hours_by_type_per_tick=gpu_hours_tick,
        migration_count=scale_events, config=cfg)
    tw = sum(weights)

    def wmean(vals):
        return sum(v * w for v, w in zip(vals, weights)) / tw if tw else 0.0

    return AblationResult(
        name=name, sizing=sizing, affinity=affinity,
        goodput_per_dollar=kpi.sla_safe_goodput_per_infra_dollar,
        sla_compliant_requests=kpi.sla_compliant_goodput,
        completed_requests=sum(tokens),
        infra_cost=kpi.total_infrastructure_cost, replica_hours=replica_hours,
        e2e_p99_s=wmean(e2e99), timeout_rate_pct=wmean(timeouts),
        mean_cold_start_s=cold_sum / cold_w if cold_w else 0.0)


def run_ablation(requests: Sequence[NormalizedGenAIRequest], *,
                 tick_seconds: float = 3600.0,
                 cold_start_s: Optional[dict] = None) -> dict:
    cold = dict(gb.DEFAULT_COLD_START)
    if cold_start_s:
        for k in ("basemodel_load", "lora_load", "controlnet_load"):
            if cold_start_s.get(k):
                cold[k] = cold_start_s[k]
    ticks = gb._aggregate_ticks(list(requests), tick_seconds)
    tick_hours = tick_seconds / 3600.0
    results = {name: _run_config(name, sizing, aff, ticks, cold, tick_hours)
               for name, sizing, aff in ABLATION_CONFIGS}
    return results


def _gpd(results, name) -> float:
    r = results.get(name)
    return (r.goodput_per_dollar or 0.0) if r else 0.0


def attribute(results: dict) -> dict:
    """Attribute the constraint_aware-vs-sla_aware gain to affinity vs sizing.

    Uses the 2×2 factorial corners (factor A = sizing: reactive_sla →
    anticipatory_sla; factor B = affinity: off → on) and a Shapley decomposition
    (average marginal contribution over both orderings), plus single-factor
    "alone" lifts measured against FIFO.
    """
    base = _gpd(results, "sla_aware")            # headline baseline
    ca = _gpd(results, "constraint_aware")       # full optimizer
    total_gain_pct = ((ca - base) / base * 100.0) if base > 0 else 0.0

    # 2x2 corners
    s0a0 = _gpd(results, "sla_aware")                    # reactive,  no affinity
    s0a1 = _gpd(results, "sla_aware_plus_affinity")      # reactive,  affinity
    s1a0 = _gpd(results, "constraint_aware_no_affinity")  # anticipatory, no aff
    s1a1 = _gpd(results, "constraint_aware")             # anticipatory, affinity

    def pct(x, y):
        return ((x - y) / y * 100.0) if y > 0 else 0.0

    # Shapley (in goodput/$ units), then normalised to % of total gain
    affinity_marg = 0.5 * ((s0a1 - s0a0) + (s1a1 - s1a0))
    sizing_marg = 0.5 * ((s1a0 - s0a0) + (s1a1 - s0a1))
    total = (s1a1 - s0a0) or 1.0
    interaction = (s1a1 - s0a0) - affinity_marg - sizing_marg

    fifo = _gpd(results, "fifo")
    return {
        "headline_baseline": "sla_aware",
        "constraint_aware_vs_sla_aware_gain_pct": round(total_gain_pct, 2),
        "single_factor_lift_vs_fifo_pct": {
            "model_affinity_alone": round(pct(_gpd(results, "fifo_plus_affinity"), fifo), 2),
            "prewarm_alone": round(pct(_gpd(results, "fifo_plus_affinity"), fifo), 2),
            "queue_awareness_alone": round(pct(_gpd(results, "queue_aware"), fifo), 2),
            "utilization_awareness_alone": round(pct(_gpd(results, "utilization_aware"), fifo), 2),
            "anticipatory_sizing_alone": round(pct(s1a0, fifo), 2),
            "combined_constraint_aware": round(pct(ca, fifo), 2),
        },
        "shapley_attribution_of_ca_vs_sla_gain": {
            "affinity_share_pct": round(100.0 * affinity_marg / total, 1),
            "sizing_share_pct": round(100.0 * sizing_marg / total, 1),
            "interaction_share_pct": round(100.0 * interaction / total, 1),
            "affinity_goodput_per_dollar": round(affinity_marg, 3),
            "sizing_goodput_per_dollar": round(sizing_marg, 3),
            "interaction_goodput_per_dollar": round(interaction, 3),
        },
        "prewarm_equals_affinity": True,
        "verdict": _verdict(affinity_marg, sizing_marg, interaction),
    }


def _verdict(affinity_marg, sizing_marg, interaction) -> str:
    parts = {"model-affinity/prewarm": affinity_marg, "anticipatory-sizing": sizing_marg,
             "interaction": interaction}
    dom = max(parts, key=parts.get)
    share = parts[dom] / (affinity_marg + sizing_marg + interaction or 1.0)
    if dom == "model-affinity/prewarm" and share >= 0.5:
        return ("primarily a model-affinity / prewarming effect "
                f"(~{round(100*share)}% of the gain); anticipatory sizing is secondary")
    return f"primarily a {dom} effect (~{round(100*share)}% of the gain)"
