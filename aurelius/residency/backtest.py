"""Per-request residency-routing backtest over the Alibaba GenAI 2026 trace.

This is the **decision-engine** companion to the tick-sizing ablation in
``aurelius/traces/genai_backtest.py`` / ``genai_ablation.py`` (which is preserved
unchanged). Where the ablation models affinity as a tick-level cold-start
amortization, this harness replays requests **one at a time** through a
simulated GPU pool and lets each policy decide *where* to route — so it can
report true per-request residency hit/miss, cold-start distribution, prewarm /
route-to-resident / eviction counts, and warm-pool GPU-hours.

It is a **simulator** (``sim.SIMULATOR_MODE``): it mutates *simulated*
``ModelLocationState`` only. No real cluster, router, or serving engine is
touched. Directional simulator/backtest evidence — **not production savings**
(``docs/RESULTS.md`` §8). The primary KPI is unchanged: SLA-safe goodput per
infrastructure dollar; residency metrics are diagnostics.

Policies compared:
  * ``fifo_round_robin``       — round-robin, residency-blind.
  * ``sla_aware_least_queue``  — least-queue, residency-blind.
  * ``sla_aware_naive_prewarm``— least-queue but everything kept warm (no cold
    starts) at an explicit warm-pool cost (one warm replica per distinct model).
  * ``affinity_only``          — route to a resident replica when one exists.
  * ``residency_engine``       — the Model Residency Decision Engine
    (``decision.choose_residency_decision`` + ``sim.apply_residency_decision``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from aurelius.benchmarks.economics import compute_sla_safe_goodput_per_infra_dollar

from .decision import SafetyContext, choose_residency_decision
from .metrics import percentile
from .models import (
    ModelLoadProfile,
    ModelLocationState,
    ModelResidencyRequest,
    ResidencyAction,
)

POLICIES = (
    "fifo_round_robin", "sla_aware_least_queue", "sla_aware_naive_prewarm",
    "affinity_only", "residency_engine",
)

# Documented priors (identical across policies). Override before any claim.
DEFAULT_GPU_MEM_GB = 80.0
DEFAULT_MODEL_MEM_GB = 16.0
DEFAULT_ADAPTER_MEM_GB = 2.0
DEFAULT_GPU_HOUR_PRICE = 3.0
DEFAULT_SLA_ABS_S = 30.0
DEFAULT_SLA_MULT = 2.0
DEFAULT_SERVICE_FLOOR_S = 0.5
WARM_POOL_HOLD_HOURS = 1.0


@dataclass
class ResidencyPolicyResult:
    policy: str
    goodput_per_dollar: Optional[float]
    sla_compliant_requests: int
    completed_requests: int
    total_requests: int
    infra_cost: float
    base_gpu_hours: float
    warm_pool_gpu_hours: float
    model_residency_hits: int
    model_residency_eligible: int
    adapter_residency_hits: int
    adapter_residency_eligible: int
    cold_start_count: int
    cold_start_p50_s: Optional[float]
    cold_start_p95_s: Optional[float]
    cold_start_p99_s: Optional[float]
    route_to_resident_count: int
    prewarm_count: int
    eviction_count: int
    sla_violations: int
    e2e_p95_s: Optional[float]
    e2e_p99_s: Optional[float]
    mean_cold_start_s: float

    @property
    def model_residency_hit_rate(self) -> Optional[float]:
        return (self.model_residency_hits / self.model_residency_eligible
                if self.model_residency_eligible else None)

    @property
    def adapter_residency_hit_rate(self) -> Optional[float]:
        return (self.adapter_residency_hits / self.adapter_residency_eligible
                if self.adapter_residency_eligible else None)

    def summary(self) -> dict:
        return {
            "policy": self.policy,
            "goodput_unit": "completed_requests",
            "sla_safe_goodput_per_infra_dollar": (
                None if self.goodput_per_dollar is None
                else round(self.goodput_per_dollar, 4)),
            "sla_compliant_requests": self.sla_compliant_requests,
            "completed_requests": self.completed_requests,
            "total_requests": self.total_requests,
            "infra_cost": round(self.infra_cost, 2),
            "base_gpu_hours": round(self.base_gpu_hours, 2),
            "warm_pool_gpu_hours": round(self.warm_pool_gpu_hours, 2),
            "model_residency_hit_rate": (
                None if self.model_residency_hit_rate is None
                else round(self.model_residency_hit_rate, 4)),
            "adapter_residency_hit_rate": (
                None if self.adapter_residency_hit_rate is None
                else round(self.adapter_residency_hit_rate, 4)),
            "cold_start_count": self.cold_start_count,
            "cold_start_p50_s": self.cold_start_p50_s,
            "cold_start_p95_s": self.cold_start_p95_s,
            "cold_start_p99_s": self.cold_start_p99_s,
            "mean_cold_start_s": round(self.mean_cold_start_s, 3),
            "route_to_resident_count": self.route_to_resident_count,
            "prewarm_count": self.prewarm_count,
            "eviction_count": self.eviction_count,
            "sla_violations": self.sla_violations,
            "e2e_latency_s_p95": self.e2e_p95_s,
            "e2e_latency_s_p99": self.e2e_p99_s,
        }


@dataclass
class _GpuSim:
    """Mutable per-GPU simulation state wrapping a ModelLocationState."""

    loc: ModelLocationState
    busy_until_s: float = 0.0
    capacity_models: int = 4
    lru: list = field(default_factory=list)   # model ids, oldest first

    def free_gb(self) -> float:
        return self.loc.memory_free_gb if self.loc.memory_free_gb is not None else 0.0

    def touch(self, model_id: str):
        if model_id in self.lru:
            self.lru.remove(model_id)
        self.lru.append(model_id)


def _build_profiles(requests, cold) -> dict:
    """One ModelLoadProfile per distinct base model (calibrated cold load)."""
    base = cold.get("basemodel_load", 22.0)
    base_p95 = base * 1.4
    lora = cold.get("lora_load", 4.4)
    profiles: dict = {}
    for r in requests:
        mid = r.service_id or "unknown"
        if mid not in profiles:
            profiles[mid] = ModelLoadProfile(
                model_id=mid, cold_load_p50_s=base, cold_load_p95_s=base_p95,
                adapter_load_p50_s=lora, adapter_load_p95_s=lora * 1.4,
                memory_required_gb=DEFAULT_MODEL_MEM_GB, source="genai_calibration",
                confidence="medium")
    return profiles


def _service_s(r) -> float:
    return max(DEFAULT_SERVICE_FLOOR_S, r.e2e_latency_s or DEFAULT_SERVICE_FLOOR_S)


def _ensure_capacity(gpu: _GpuSim, need_gb: float, protect: str) -> int:
    """Evict LRU models until ``need_gb`` fits. Returns evictions performed."""
    evictions = 0
    while (gpu.loc.gpu_memory_total is not None
           and gpu.free_gb() < need_gb and gpu.loc.loaded_model_ids):
        victim = None
        for m in gpu.lru:
            if m != protect:
                victim = m
                break
        if victim is None:
            break
        gpu.loc.loaded_model_ids.remove(victim)
        gpu.lru.remove(victim)
        if gpu.loc.gpu_memory_used is not None:
            gpu.loc.gpu_memory_used = max(
                0.0, gpu.loc.gpu_memory_used - DEFAULT_MODEL_MEM_GB * 1e9)
        evictions += 1
    return evictions


def _load_model(gpu: _GpuSim, model_id: str) -> int:
    """Load a model onto a GPU (evicting LRU if needed). Returns evictions."""
    ev = _ensure_capacity(gpu, DEFAULT_MODEL_MEM_GB, protect=model_id)
    if model_id not in gpu.loc.loaded_model_ids:
        gpu.loc.loaded_model_ids.append(model_id)
        if gpu.loc.gpu_memory_used is not None:
            gpu.loc.gpu_memory_used += DEFAULT_MODEL_MEM_GB * 1e9
    gpu.touch(model_id)
    return ev


def _make_cluster(n_gpus: int, gpu_mem_gb: float) -> list:
    gpus = []
    cap = max(1, int(gpu_mem_gb // DEFAULT_MODEL_MEM_GB))
    for i in range(n_gpus):
        loc = ModelLocationState(
            region="sim", node_id=f"node-{i // 8}", gpu_id=f"gpu-{i}",
            container_id=f"pod-{i}", loaded_model_ids=[], loaded_adapter_ids=[],
            gpu_memory_used=0.0, gpu_memory_total=gpu_mem_gb * 1e9,
            estimated_queue_wait_s=0.0, telemetry_confidence="high")
        gpus.append(_GpuSim(loc=loc, capacity_models=cap))
    return gpus


def _run_policy(policy: str, requests, cold, *, n_gpus, gpu_mem_gb,
                gpu_hour_price, sla_abs_s, sla_mult, profiles) -> ResidencyPolicyResult:
    gpus = _make_cluster(n_gpus, gpu_mem_gb)
    by_key = {g.loc.location_key: g for g in gpus}
    base_load = cold.get("basemodel_load", 22.0)
    lora_load = cold.get("lora_load", 4.4)

    ctx = SafetyContext(
        gpu_hour_price=gpu_hour_price, default_latency_sla_ms=1e9,
        service_time_proxy_s=DEFAULT_SERVICE_FLOOR_S, min_telemetry_confidence="low",
        warm_pool_hold_hours=WARM_POOL_HOLD_HOURS, prewarm_expected_hits=10.0)

    cold_starts: list = []
    n_cold = n_model_hit = n_model_elig = 0
    n_adapter_hit = n_adapter_elig = 0
    n_route_resident = n_prewarm = n_evict = 0
    e2es: list = []
    sla_compliant = sla_violations = 0

    times = [r.timestamp_s for r in requests if r.timestamp_s is not None]
    t0 = min(times) if times else 0.0
    t1 = max(times) if times else 0.0
    duration_h = max((t1 - t0) / 3600.0, 1e-9)

    distinct_models = len({r.service_id or "unknown" for r in requests})

    rr = 0
    for r in sorted(requests, key=lambda x: (x.timestamp_s or 0.0, x.request_id)):
        t = r.timestamp_s or 0.0
        model_id = r.service_id or "unknown"
        adapter_id = f"lora:{model_id}" if (r.num_lora or 0) > 0 else None
        service = _service_s(r)
        sla = sla_abs_s + sla_mult * service
        naive = policy == "sla_aware_naive_prewarm"

        # refresh queue-wait estimates for all gpus at this arrival
        for g in gpus:
            g.loc.estimated_queue_wait_s = max(0.0, g.busy_until_s - t)

        # --- choose target gpu by policy (engine drives routing) ---
        if policy == "fifo_round_robin":
            target = gpus[rr % n_gpus]
            rr += 1
        elif policy in ("sla_aware_least_queue", "sla_aware_naive_prewarm"):
            target = min(gpus, key=lambda g: g.busy_until_s)
        elif policy == "affinity_only":
            resident = [g for g in gpus if model_id in g.loc.loaded_model_ids]
            target = (min(resident, key=lambda g: g.busy_until_s) if resident
                      else min(gpus, key=lambda g: g.busy_until_s))
        elif policy == "residency_engine":
            # stateless per-request routing: let the engine pick the best warm
            # replica by KPI each request (residency = any resident replica). The
            # finer-grained current_route affinity-preservation path is exercised
            # by the unit tests, not forced here.
            req = ModelResidencyRequest(
                request_id=r.request_id, timestamp=t, workload_id=model_id,
                model_id=model_id, adapter_id=adapter_id, priority_class="standard",
                latency_sla_ms=sla * 1000.0, region="sim", current_route=None)
            decision = choose_residency_decision(
                req, [g.loc for g in gpus], profiles, ctx, ctx)
            if decision.action == ResidencyAction.ROUTE_TO_RESIDENT_MODEL:
                n_route_resident += 1
            elif decision.action in (ResidencyAction.PREWARM_MODEL,
                                     ResidencyAction.PREWARM_ADAPTER):
                n_prewarm += 1
            key = decision.target_location or decision.current_location
            target = by_key.get(key) or min(gpus, key=lambda g: g.busy_until_s)
        else:  # pragma: no cover
            raise ValueError(policy)

        # --- residency hit/miss measured BEFORE any load at the chosen gpu ---
        model_resident = naive or (model_id in target.loc.loaded_model_ids)
        n_model_elig += 1
        cold_load = 0.0
        if model_resident:
            n_model_hit += 1
        else:
            cold_load += base_load
        if adapter_id is not None:
            n_adapter_elig += 1
            adapter_resident = naive or (adapter_id in target.loc.loaded_adapter_ids)
            if adapter_resident:
                n_adapter_hit += 1
            else:
                cold_load += lora_load
        if cold_load > 0:
            n_cold += 1
            cold_starts.append(cold_load)

        # --- mutate simulated state uniformly (load on miss, evict LRU if full) ---
        if not naive:
            n_evict += _load_model(target, model_id)
            if adapter_id is not None and adapter_id not in target.loc.loaded_adapter_ids:
                target.loc.loaded_adapter_ids.append(adapter_id)

        queue_wait = max(0.0, target.busy_until_s - t)
        e2e = queue_wait + cold_load + service
        target.busy_until_s = max(target.busy_until_s, t) + service + cold_load

        e2es.append(e2e)
        if e2e <= sla:
            sla_compliant += 1
        else:
            sla_violations += 1

    # --- cost basis ---
    # All routing policies share the SAME fixed base pool (same denominator);
    # they differ only in SLA-safe goodput (numerator) via cold-start latency.
    # naive_prewarm_all keeps EVERY distinct model warm simultaneously, so it
    # pays for the replicas it must hold BEYOND the base pool's capacity — the
    # honest extra cost of "keep everything warm" (zero only when all models fit).
    base_gpu_hours = n_gpus * duration_h
    pool_capacity = n_gpus * max(1, int(gpu_mem_gb // DEFAULT_MODEL_MEM_GB))
    if policy == "sla_aware_naive_prewarm":
        extra_replicas = max(0.0, distinct_models - pool_capacity)
        warm_pool_gpu_hours = extra_replicas * duration_h
    else:
        warm_pool_gpu_hours = 0.0
    infra_cost = (base_gpu_hours + warm_pool_gpu_hours) * gpu_hour_price
    gpd = compute_sla_safe_goodput_per_infra_dollar(sla_compliant, infra_cost)

    return ResidencyPolicyResult(
        policy=policy, goodput_per_dollar=gpd, sla_compliant_requests=sla_compliant,
        completed_requests=len(requests), total_requests=len(requests),
        infra_cost=infra_cost, base_gpu_hours=base_gpu_hours,
        warm_pool_gpu_hours=warm_pool_gpu_hours,
        model_residency_hits=n_model_hit, model_residency_eligible=n_model_elig,
        adapter_residency_hits=n_adapter_hit, adapter_residency_eligible=n_adapter_elig,
        cold_start_count=n_cold,
        cold_start_p50_s=percentile(cold_starts, 50) if cold_starts else None,
        cold_start_p95_s=percentile(cold_starts, 95) if cold_starts else None,
        cold_start_p99_s=percentile(cold_starts, 99) if cold_starts else None,
        route_to_resident_count=n_route_resident, prewarm_count=n_prewarm,
        eviction_count=n_evict, sla_violations=sla_violations,
        e2e_p95_s=round(percentile(e2es, 95), 3) if e2es else None,
        e2e_p99_s=round(percentile(e2es, 99), 3) if e2es else None,
        mean_cold_start_s=(sum(cold_starts) / len(cold_starts)) if cold_starts else 0.0)


@dataclass
class ResidencyBacktestResult:
    n_requests: int
    n_gpus: int
    cold_start_s: dict
    policy_results: dict

    def to_summary_dict(self) -> dict:
        return {
            "primary_kpi": "sla_safe_goodput_per_infrastructure_dollar",
            "goodput_unit": "completed_requests",
            "directional_only_not_production_savings": True,
            "harness": "per_request_residency_routing (simulator mode)",
            "n_requests": self.n_requests, "n_gpus": self.n_gpus,
            "cold_start_calibration_s": self.cold_start_s,
            "policies": {p: r.summary() for p, r in self.policy_results.items()},
        }


def run_residency_backtest(requests, *, cold_start_s: Optional[dict] = None,
                           n_gpus: int = 8, gpu_mem_gb: float = DEFAULT_GPU_MEM_GB,
                           gpu_hour_price: float = DEFAULT_GPU_HOUR_PRICE,
                           sla_abs_s: float = DEFAULT_SLA_ABS_S,
                           sla_mult: float = DEFAULT_SLA_MULT,
                           policies=POLICIES) -> ResidencyBacktestResult:
    cold = {"basemodel_load": 22.0, "lora_load": 4.4, "controlnet_load": 3.9}
    if cold_start_s:
        for k in ("basemodel_load", "lora_load", "controlnet_load"):
            if cold_start_s.get(k):
                cold[k] = cold_start_s[k]
    requests = list(requests)
    profiles = _build_profiles(requests, cold)
    results = {p: _run_policy(p, requests, cold, n_gpus=n_gpus, gpu_mem_gb=gpu_mem_gb,
                              gpu_hour_price=gpu_hour_price, sla_abs_s=sla_abs_s,
                              sla_mult=sla_mult, profiles=profiles)
               for p in policies}
    return ResidencyBacktestResult(
        n_requests=len(requests), n_gpus=n_gpus, cold_start_s=cold,
        policy_results=results)
