"""Discrete-event cluster simulator engine for constraint-aware Aurelius.

Produces ClusterState snapshots tick-by-tick that pass through the same
connector interfaces as real customer telemetry. Aurelius cannot tell whether
it is connected to this simulator or a real cluster at the connector boundary.

Physical models are proxies, not physics simulations:
- Thermal: low-pass filtered temperature, throttle when temp > threshold
- Queue/latency: M/M/1 approximation with diurnal modulation
- KV cache: memory-pressure proxy
- Communication: topology link type penalty

All outputs carry is_sandbox=True and are excluded from any economic claims.
"""

from __future__ import annotations

import math
import random
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from ...state.models import (
    ClusterState,
    EnergyState,
    GPUState,
    InferenceServiceState,
    NodeState,
    Provenance,
    RegionState,
    TopologyLinkType,
    TopologyState,
)
from . import kv_cache as kvc
from . import migration as mig
from . import serving
from . import thermal as therm
from .cache_model import WorkloadCacheState
from .calibration import (
    kv_value,
    migration_value,
    power_class_for_model,
    serving_value,
    thermal_value,
)
from .migration_model import WorkloadMigrationState
from .model import (
    GPU_PROFILES,
    GPUProfile,
    SimCluster,
    SimGPU,
    SimNode,
    SimQueue,
    SimRegion,
    SimTopologyLink,
    SimulatorConfig,
    SimWorkload,
)
from .thermal_model import GPUThermalState, RackThermalState

# EnergyState and ThermalState are used for region-level context
# (not directly emitted per-GPU in this version)

# ---------------------------------------------------------------------------
# Simulator constants
# ---------------------------------------------------------------------------

_THERMAL_ALPHA = 0.25       # low-pass filter for temperature EMA
_RACK_HEAT_ALPHA = 0.15     # rack heat accumulation rate
_RACK_HEAT_DECAY = 0.05     # rack heat dissipation per tick at low load
_THROTTLE_TEMP_C = 83.0
_MAX_REALISTIC_TEMP_C = 100.0

# M/M/1 queue: latency scales as 1/(1 - rho) where rho = lambda/mu
_BASE_TTFT_MS = 150.0       # TTFT at zero load
_BASE_TPOT_MS = 20.0        # TPOT per token at zero load

_TOKENS_PER_REQUEST = 256   # average tokens per request (proxy)
_SLA_P99_DEFAULT_MS = 2000.0

# Cold start penalty: workload needs warmup_ticks before reaching full perf
_COLD_START_WARMUP_TICKS = 2


# ---------------------------------------------------------------------------
# Simulator tick result
# ---------------------------------------------------------------------------

@dataclass
class TickMetrics:
    """Per-tick aggregated metrics for comparison reports."""
    tick: int
    timestamp: datetime
    total_energy_cost: float             # $ for this tick
    total_tokens: int                    # tokens served this tick
    total_energy_kwh: float              # kWh used
    cost_per_token: Optional[float]      # $/token (None if no tokens)
    tokens_per_joule: Optional[float]    # tokens/J (None if no energy)
    mean_gpu_util_pct: float
    p95_latency_ms: Optional[float]
    p99_latency_ms: Optional[float]
    queue_wait_p95_ms: Optional[float]
    sla_violations: int
    thermal_throttle_gpu_count: int
    migration_count: int
    mean_topology_score: float
    # KV-cache / prefix-affinity / locality realism KPIs
    kv_pressure_max: Optional[float] = None
    prefix_hit_rate_mean: Optional[float] = None
    preemption_count: int = 0
    recompute_count: int = 0
    cold_reroute_count: int = 0
    cache_eviction_count: int = 0
    locality_confidence_mean: Optional[float] = None
    cache_fragmentation_frac_mean: Optional[float] = None
    routing_affinity_score_mean: Optional[float] = None
    ttft_p50_ms: Optional[float] = None
    ttft_p95_ms: Optional[float] = None
    ttft_p99_ms: Optional[float] = None
    # Migration / rerouting / drain / cold-start realism KPIs
    reroute_count: int = 0
    migration_veto_count: int = 0
    drain_seconds_total: float = 0.0
    startup_latency_s_max: Optional[float] = None
    warmup_active_count: int = 0
    batch_efficiency_mean: Optional[float] = None
    route_churn_mean: Optional[float] = None
    proxy_saturation_max: Optional[float] = None
    cold_start_count: int = 0
    warm_pool_occupancy_mean: Optional[float] = None
    rollback_count: int = 0
    overload_events: int = 0
    # Thermal / cooling / power realism KPIs
    max_gpu_temp_c: Optional[float] = None
    max_rack_inlet_c: Optional[float] = None
    thermal_slowdown_pct_mean: Optional[float] = None
    power_slowdown_pct_mean: Optional[float] = None
    thermal_throttle_events: int = 0
    hotspot_severity_max: Optional[float] = None
    rack_density_kw_max: Optional[float] = None
    thermal_excursions: int = 0
    cooling_alarms: int = 0
    thermal_migration_vetoes: int = 0
    is_sandbox: bool = True


@dataclass
class SimulatorTick:
    """Result of a single simulator tick."""
    tick: int
    timestamp: datetime
    cluster_state: ClusterState
    metrics: TickMetrics
    dcgm_texts: dict[str, str]           # node_id → Prometheus DCGM text
    vllm_texts: dict[str, str]           # service_id → Prometheus vLLM text
    k8s_node_list: dict[str, Any]
    k8s_pod_list: dict[str, Any]
    topology_texts: dict[str, str]       # node_id → nvidia-smi topo text


# ---------------------------------------------------------------------------
# ClusterSimulator
# ---------------------------------------------------------------------------

class ClusterSimulator:
    """Discrete-event hourly-tick GPU cluster simulator.

    Provides:
    - get_cluster_state() → canonical ClusterState
    - get_dcgm_prometheus_text(node_id) → DCGM Prometheus text
    - get_vllm_prometheus_text(service_id) → vLLM Prometheus text
    - get_kubernetes_node_list() → fake V1NodeList dict
    - get_kubernetes_pod_list() → fake V1PodList dict
    - get_nvidia_smi_topo_text(node_id) → fake nvidia-smi topo -m text
    """

    def __init__(self, config: SimulatorConfig, seed: Optional[int] = None):
        self.config = config
        self.seed = seed if seed is not None else config.seed
        self._serving_config = dict(getattr(config, "serving_config", {}) or {})
        self._rng = random.Random(self.seed)
        self._cluster = self._build_initial_cluster()
        self._base_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        self._tick_metrics: list[TickMetrics] = []

    def reset(self) -> None:
        """Reset to initial state (same seed → identical replay)."""
        self._rng = random.Random(self.seed)
        self._cluster = self._build_initial_cluster()
        self._tick_metrics = []

    # ------------------------------------------------------------------
    # Build initial state from config
    # ------------------------------------------------------------------

    def _build_initial_cluster(self) -> SimCluster:
        cluster = SimCluster(tick_duration_hours=self.config.tick_duration_hours)

        for r_cfg in self.config.regions:
            region = self._build_region(r_cfg)
            cluster.regions[region.region_id] = region

        for w_cfg in self.config.workloads:
            workload = self._build_workload(w_cfg, cluster)
            cluster.workloads[workload.workload_id] = workload

        # Compute initial topology scores
        for workload in cluster.workloads.values():
            workload.topology_score = self._compute_topology_score(workload, cluster)

        return cluster

    @staticmethod
    def _parse_float_trace(raw: list) -> list[float]:
        """Parse a trace that may contain ints, floats, or dash-separated strings.

        YAML files sometimes encode multiple values as ``"200 - 210 - 220"``.
        This helper handles that encoding alongside normal int/float list items.
        """
        result: list[float] = []
        for item in raw:
            if isinstance(item, str):
                for token in item.split(" - "):
                    token = token.strip()
                    if token:
                        result.append(float(token))
            else:
                result.append(float(item))
        return result

    def _build_region(self, r_cfg: dict[str, Any]) -> SimRegion:
        region_id = r_cfg["region_id"]
        raw_carbon = r_cfg.get("carbon_intensity_trace", [])
        raw_ambient = r_cfg.get("ambient_temp_trace", [])
        region = SimRegion(
            region_id=region_id,
            energy_price_trace=self._parse_float_trace(r_cfg.get("energy_price_trace", [50.0])),
            carbon_intensity_trace=self._parse_float_trace(raw_carbon),
            current_energy_price=self._parse_float_trace(r_cfg.get("energy_price_trace", [50.0]))[0],
            ambient_temp_c=float(r_cfg.get("ambient_temp_c", 22.0)),
            ambient_temp_trace=self._parse_float_trace(raw_ambient),
            network_latency_to=r_cfg.get("network_latency_to", {}),
        )

        for node_cfg in r_cfg.get("nodes", []):
            node = self._build_node(node_cfg, region_id)
            region.nodes.append(node)

        for q_cfg in r_cfg.get("queues", []):
            queue = SimQueue(
                queue_id=q_cfg.get("queue_id", f"{region_id}-q{len(region.queues)}"),
                service_id=q_cfg["service_id"],
                region_id=region_id,
                base_arrival_rate_per_sec=q_cfg.get("base_arrival_rate_per_sec", 1.0),
                diurnal_amplitude=q_cfg.get("diurnal_amplitude", 0.4),
                queue_depth=0,
            )
            region.queues.append(queue)

        # Compute capacities
        region.total_gpus = sum(n.gpu_count for n in region.nodes)
        region.available_gpus = region.total_gpus

        return region

    def _build_node(self, node_cfg: dict[str, Any], region_id: str) -> SimNode:
        node_id = node_cfg["node_id"]
        gpu_type = node_cfg.get("gpu_type", "a100-sxm4-80gb")
        gpu_count = node_cfg.get("gpu_count", 4)
        # "nvswitch", "pcie", "pcie_multi_numa"
        topology_class = node_cfg.get("topology_class", "nvswitch")
        rack_id = node_cfg.get("rack_id", f"{region_id}-rack0")
        zone = node_cfg.get("zone", f"{region_id}-zone-a")

        profile = GPU_PROFILES.get(gpu_type, GPU_PROFILES["a100-sxm4-80gb"])

        power_class = power_class_for_model(profile.model_name)
        gpus = []
        for idx in range(gpu_count):
            gpu_id = f"{node_id}-gpu{idx}"
            gpu_uuid = f"GPU-{uuid.uuid4().hex[:8].upper()}"
            t0 = self._rng.uniform(32.0, 38.0)
            gpu = SimGPU(
                gpu_id=gpu_id,
                gpu_index=idx,
                uuid=gpu_uuid,
                node_id=node_id,
                profile=profile,
                temperature_c=t0,
                power_watts=profile.base_power_watts * 0.1,  # idle
                power_cap_watts=power_class.p_max_w,
            )
            # First-class per-GPU thermal state.
            gt = GPUThermalState()
            gt.inertia.temp_c = t0
            gt.power_class = power_class.name
            gt.power_throttle.power_cap_w = power_class.p_max_w
            gpu.thermal = gt
            gpus.append(gpu)

        # Build topology links
        links = self._build_topology_links(gpus, topology_class)

        cooling_regime = node_cfg.get("cooling_regime", "air")
        node = SimNode(
            node_id=node_id,
            region_id=region_id,
            zone=zone,
            rack_id=rack_id,
            instance_type=node_cfg.get("instance_type", f"gpu.{gpu_count}x{gpu_type}"),
            gpus=gpus,
            cooling_regime=cooling_regime,
            labels={
                "topology.kubernetes.io/region": region_id,
                "topology.kubernetes.io/zone": zone,
                "gpu-type": gpu_type,
                "topology-class": topology_class,
                "cooling-regime": cooling_regime,
            },
            topology_links=links,
        )
        # First-class per-rack thermal state.
        rt = RackThermalState()
        rt.cooling_regime = cooling_regime
        rt.zone.regime = cooling_regime
        node.rack_thermal = rt
        return node

    def _build_topology_links(
        self, gpus: list[SimGPU], topology_class: str
    ) -> list[SimTopologyLink]:
        links = []
        n = len(gpus)

        if topology_class == "nvswitch":
            # All GPUs fully connected via NVSwitch (H100 DGX style)
            link_type = "NVSWITCH"
            bw = gpus[0].profile.nvlink_bandwidth_gbps if gpus else 0.0
        elif topology_class in ("nvlink4", "nvlink2"):
            link_type = "NV4" if topology_class == "nvlink4" else "NV2"
            bw = gpus[0].profile.nvlink_bandwidth_gbps * 0.5 if gpus else 0.0
        elif topology_class == "pcie_multi_numa":
            # NUMA-aware: within NUMA fast (PIX), cross-NUMA slow (PHB)
            link_type = None   # set per-pair
            bw = gpus[0].profile.pcie_bandwidth_gbps if gpus else 0.0
        else:
            # All PCIe
            link_type = "PIX"
            bw = gpus[0].profile.pcie_bandwidth_gbps if gpus else 0.0

        for i in range(n):
            for j in range(i + 1, n):
                if topology_class == "pcie_multi_numa":
                    # GPUs 0..n//2-1 on NUMA0, n//2..n-1 on NUMA1
                    same_numa = (i < n // 2) == (j < n // 2)
                    lt = "PIX" if same_numa else "PHB"
                    link_bw = bw * (1.0 if same_numa else 0.5)
                else:
                    lt = link_type
                    link_bw = bw

                links.append(SimTopologyLink(
                    gpu_a=gpus[i].gpu_id,
                    gpu_b=gpus[j].gpu_id,
                    link_type=lt,
                    bandwidth_gbps=link_bw,
                ))

        return links

    def _build_workload(self, w_cfg: dict[str, Any], cluster: SimCluster) -> SimWorkload:
        workload_id = w_cfg.get("workload_id", f"wl-{uuid.uuid4().hex[:8]}")
        region_id = w_cfg.get("region_id", next(iter(cluster.regions), "us-east"))

        workload = SimWorkload(
            workload_id=workload_id,
            service_id=w_cfg.get("service_id", workload_id),
            workload_type=w_cfg.get("workload_type", "inference"),
            priority_tier=w_cfg.get("priority_tier", "standard"),
            region_id=region_id,
            gpu_count_required=w_cfg.get("gpu_count_required", 1),
            target_util_pct=w_cfg.get("target_util_pct", 60.0),
            memory_required_bytes=w_cfg.get("memory_required_bytes", 20 * 1024**3),
            communication_intensity=w_cfg.get("communication_intensity", "low"),
            memory_intensity=w_cfg.get("memory_intensity", "medium"),
            latency_sensitive=w_cfg.get("latency_sensitive", False),
            sla_policy_id=w_cfg.get("sla_policy_id"),
            latency_sla_p99_ms=w_cfg.get("latency_sla_p99_ms"),
            queue_sla_p95_ms=w_cfg.get("queue_sla_p95_ms"),
            migration_allowed=w_cfg.get("migration_allowed", True),
            kv_cache_usage_frac=w_cfg.get("kv_cache_usage_frac", 0.3),
            prefix_cache_hit_rate_frac=w_cfg.get("prefix_cache_hit_rate_frac", 0.5),
            model_kv_profile=w_cfg.get("model_kv_profile", "llama3-8b"),
            prefix_overlap=w_cfg.get("prefix_overlap", 0.5),
            avg_seq_len_tokens=w_cfg.get("avg_seq_len_tokens", 1024),
            engine_runtime=w_cfg.get("engine_runtime", "vllm"),
            warm_pool_size=w_cfg.get("warm_pool_size", 0),
            pdb_min_available=w_cfg.get("pdb_min_available", 0),
        )

        # Place workload onto GPUs in the target region
        self._place_workload(workload, cluster)

        # Initialize first-class KV-cache / locality state on its home route.
        cfg = self._serving_config or None
        cache = WorkloadCacheState()
        cache.locality.confidence = kv_value("locality_confidence_init", cfg)
        cache.prefix.overlap = workload.prefix_overlap
        cache.prefix.shared_prefix_tokens = (
            workload.prefix_overlap * workload.avg_seq_len_tokens
        )
        cache.routing.home_region = workload.region_id
        cache.routing.home_gpu_ids = tuple(workload.gpu_ids)
        cache.routing.telemetry_tier = (cfg or {}).get("kv_telemetry_tier", "high")
        workload.cache = cache

        # Initialize first-class migration / drain / cold-start state. The
        # workload starts WARM on its home route (no startup penalty).
        migstate = WorkloadMigrationState(engine_runtime=workload.engine_runtime)
        migstate.warm_pool.size = workload.warm_pool_size
        migstate.pdb.min_available = workload.pdb_min_available
        migstate.pdb.available = max(0, len(workload.gpu_ids) - workload.pdb_min_available)
        workload.migration = migstate
        return workload

    def _place_workload(self, workload: SimWorkload, cluster: SimCluster) -> None:
        """Assign a workload to available GPUs in its region."""
        region = cluster.regions.get(workload.region_id)
        if region is None:
            return

        assigned_gpus: list[SimGPU] = []
        assigned_nodes: set[str] = set()

        for node in region.nodes:
            for gpu in node.gpus:
                can_assign = len(assigned_gpus) < workload.gpu_count_required
                if gpu.assigned_workload_id is None and can_assign:
                    gpu.assigned_workload_id = workload.workload_id
                    gpu.memory_used_bytes = workload.memory_required_bytes
                    assigned_gpus.append(gpu)
                    assigned_nodes.add(node.node_id)
                if len(assigned_gpus) >= workload.gpu_count_required:
                    break
            if len(assigned_gpus) >= workload.gpu_count_required:
                break

        workload.gpu_ids = [g.gpu_id for g in assigned_gpus]
        workload.node_ids = list(assigned_nodes)

    # ------------------------------------------------------------------
    # Tick advancement
    # ------------------------------------------------------------------

    def tick(self) -> SimulatorTick:
        """Advance one time step and return tick outputs."""
        cluster = self._cluster
        cluster.tick += 1

        self._apply_events(cluster)
        self._update_energy_prices(cluster)
        self._update_workload_targets(cluster)
        self._update_gpu_state(cluster)
        self._update_thermal(cluster)
        self._update_kv_cache(cluster)
        self._update_migration(cluster)
        self._update_queues(cluster)
        self._update_cost_accounting(cluster)

        ts = self._tick_timestamp(cluster.tick)
        cs = self.get_cluster_state()
        metrics = self._compute_tick_metrics(cluster, ts)
        self._tick_metrics.append(metrics)

        return SimulatorTick(
            tick=cluster.tick,
            timestamp=ts,
            cluster_state=cs,
            metrics=metrics,
            dcgm_texts={
                node.node_id: self.get_dcgm_prometheus_text(node.node_id)
                for region in cluster.regions.values()
                for node in region.nodes
            },
            vllm_texts={
                wl.service_id: self.get_vllm_prometheus_text(wl.service_id)
                for wl in cluster.workloads.values()
                if wl.workload_type in ("inference", "embedding")
            },
            k8s_node_list=self.get_kubernetes_node_list(),
            k8s_pod_list=self.get_kubernetes_pod_list(),
            topology_texts={
                node.node_id: self.get_nvidia_smi_topo_text(node.node_id)
                for region in cluster.regions.values()
                for node in region.nodes
            },
        )

    def run(self, steps: int) -> list[SimulatorTick]:
        """Run for N ticks and return all tick results."""
        return [self.tick() for _ in range(steps)]

    def run_metrics_only(self, steps: int) -> list[TickMetrics]:
        """Lightweight run returning only per-tick metrics (no connector texts)."""
        ticks = []
        for _ in range(steps):
            cluster = self._cluster
            cluster.tick += 1
            self._apply_events(cluster)
            self._update_energy_prices(cluster)
            self._update_workload_targets(cluster)
            self._update_gpu_state(cluster)
            self._update_thermal(cluster)
            self._update_kv_cache(cluster)
            self._update_migration(cluster)
            self._update_queues(cluster)
            self._update_cost_accounting(cluster)
            ts = self._tick_timestamp(cluster.tick)
            m = self._compute_tick_metrics(cluster, ts)
            self._tick_metrics.append(m)
            ticks.append(m)
        return ticks

    # ------------------------------------------------------------------
    # Internal simulation updates
    # ------------------------------------------------------------------

    def _tick_timestamp(self, tick: int) -> datetime:
        return self._base_time + timedelta(hours=(tick - 1) * self._cluster.tick_duration_hours)

    def _apply_events(self, cluster: SimCluster) -> None:
        """Apply scenario events scheduled at this tick."""
        tick = cluster.tick
        for event in self.config.events:
            if event.get("tick") != tick:
                continue
            etype = event.get("type", "")

            if etype == "queue_surge":
                region_id = event.get("region_id")
                service_id = event.get("service_id")
                multiplier = event.get("multiplier", 3.0)
                for region in cluster.regions.values():
                    if region_id and region.region_id != region_id:
                        continue
                    for q in region.queues:
                        if service_id and q.service_id != service_id:
                            continue
                        q.surge_active = True
                        q.surge_multiplier = multiplier

            elif etype == "queue_surge_end":
                region_id = event.get("region_id")
                service_id = event.get("service_id")
                for region in cluster.regions.values():
                    if region_id and region.region_id != region_id:
                        continue
                    for q in region.queues:
                        if service_id and q.service_id != service_id:
                            continue
                        q.surge_active = False
                        q.surge_multiplier = 1.0

            elif etype == "energy_price_spike":
                region_id = event.get("region_id")
                multiplier = event.get("multiplier", 2.0)
                for region in cluster.regions.values():
                    if region_id and region.region_id != region_id:
                        continue
                    # Multiply from the base trace price at this tick (not cumulative)
                    if region.energy_price_trace:
                        base_idx = (tick - 1) % len(region.energy_price_trace)
                        base_price = region.energy_price_trace[base_idx]
                    else:
                        base_price = region.current_energy_price
                    region.current_energy_price = base_price * multiplier
                    region.price_spike_active = True

            elif etype == "energy_price_spike_end":
                region_id = event.get("region_id")
                # Restore from trace at current tick
                for region in cluster.regions.values():
                    if region_id and region.region_id != region_id:
                        continue
                    region.price_spike_active = False
                    if region.energy_price_trace:
                        idx = (tick - 1) % len(region.energy_price_trace)
                        region.current_energy_price = region.energy_price_trace[idx]

            elif etype == "thermal_hotspot":
                node_id = event.get("node_id")
                extra_heat = event.get("extra_heat_c", 15.0)
                for region in cluster.regions.values():
                    for node in region.nodes:
                        if node_id and node.node_id != node_id:
                            continue
                        # Event-driven inlet heat (e.g. cooling fault); added to
                        # inlet in _update_thermal and decayed gradually.
                        node.event_heat_c += extra_heat

            elif etype == "thermal_hotspot_end":
                node_id = event.get("node_id")
                for region in cluster.regions.values():
                    for node in region.nodes:
                        if node_id and node.node_id != node_id:
                            continue
                        node.event_heat_c = 0.0

            elif etype == "workload_util_change":
                workload_id = event.get("workload_id")
                new_util = event.get("target_util_pct")
                for wl in cluster.workloads.values():
                    if workload_id and wl.workload_id != workload_id:
                        continue
                    if new_util is not None:
                        wl.target_util_pct = new_util

            elif etype == "kv_cache_pressure":
                service_id = event.get("service_id")
                cache_usage = event.get("kv_cache_usage_frac", 0.9)
                hit_rate = event.get("prefix_cache_hit_rate_frac", 0.1)
                for wl in cluster.workloads.values():
                    if service_id and wl.service_id != service_id:
                        continue
                    # Force a KV-pressure floor and a depressed prefix hit rate;
                    # the KV realism layer honours these overrides each tick.
                    wl.kv_pressure_override = cache_usage
                    wl.prefix_hit_override = hit_rate

            elif etype == "kv_cache_pressure_end":
                service_id = event.get("service_id")
                for wl in cluster.workloads.values():
                    if service_id and wl.service_id != service_id:
                        continue
                    wl.kv_pressure_override = None
                    wl.prefix_hit_override = None

    def _update_energy_prices(self, cluster: SimCluster) -> None:
        tick = cluster.tick
        for region in cluster.regions.values():
            if region.energy_price_trace:
                idx = (tick - 1) % len(region.energy_price_trace)
                # Only update from trace when no spike event is overriding the price
                if not region.price_spike_active:
                    region.current_energy_price = region.energy_price_trace[idx]
            if region.carbon_intensity_trace:
                idx = (tick - 1) % len(region.carbon_intensity_trace)
                region.current_carbon_intensity = region.carbon_intensity_trace[idx]
            if region.ambient_temp_trace:
                idx = (tick - 1) % len(region.ambient_temp_trace)
                region.ambient_temp_c = region.ambient_temp_trace[idx]

    def _update_workload_targets(self, cluster: SimCluster) -> None:
        """Decrement cold-start warmup counters."""
        for wl in cluster.workloads.values():
            if wl.cold_start_warmup_ticks_remaining > 0:
                wl.cold_start_warmup_ticks_remaining -= 1

    def _update_gpu_state(self, cluster: SimCluster) -> None:
        """Update GPU utilization, power, and memory for each GPU."""
        for region in cluster.regions.values():
            for node in region.nodes:
                for gpu in node.gpus:
                    wid = gpu.assigned_workload_id
                    workload = cluster.workloads.get(wid) if wid else None

                    if workload is None:
                        # Idle GPU
                        target_util = self._rng.uniform(0.5, 2.0)   # idle noise
                    else:
                        # Workload target util with cold-start penalty
                        target_util = workload.target_util_pct
                        if workload.cold_start_warmup_ticks_remaining > 0:
                            warmup_frac = (
                                1.0
                                - workload.cold_start_warmup_ticks_remaining
                                / _COLD_START_WARMUP_TICKS
                            )
                            target_util *= warmup_frac

                    # Add small noise
                    noise = self._rng.gauss(0, 2.0)
                    util = max(0.0, min(100.0, target_util + noise))

                    gpu.utilization_pct = util
                    gpu.sm_activity_pct = util * self._rng.uniform(0.9, 1.0)

                    # Board power: saturating curve by GPU class + workload kind
                    # (utilization alone does NOT linearly predict heat).
                    pclass = therm.power_class_for_model(gpu.profile.model_name)
                    wkind = workload.workload_type if workload is not None else "inference"
                    gpu.power_watts = therm.board_power_watts(
                        util / 100.0, pclass, wkind, self._serving_config or None
                    )
                    gpu.power_cap_watts = pclass.p_max_w
                    if gpu.thermal is not None:
                        gpu.thermal.board_power_w = gpu.power_watts

                    # Memory usage
                    if workload is not None:
                        # Some variation around required amount
                        gpu.memory_used_bytes = int(
                            workload.memory_required_bytes * self._rng.uniform(0.95, 1.05)
                        )
                        gpu.memory_used_bytes = min(
                            gpu.memory_used_bytes, gpu.profile.memory_total_bytes
                        )
                    else:
                        # system footprint
                        gpu.memory_used_bytes = int(gpu.profile.memory_total_bytes * 0.01)

                    # Communication traffic (proxy)
                    if workload is not None and workload.communication_intensity != "low":
                        comm_mult = {"medium": 1.0, "high": 3.0}.get(
                            workload.communication_intensity, 0.0
                        )
                        bw = gpu.profile.nvlink_bandwidth_gbps
                        if bw > 0:
                            base_tx = bw * 1e9 * (util / 100.0) * comm_mult * 0.3
                            gpu.nvlink_tx_bytes_per_sec = base_tx * self._rng.uniform(0.9, 1.1)
                            gpu.nvlink_rx_bytes_per_sec = base_tx * self._rng.uniform(0.9, 1.1)
                        else:
                            base_tx = (
                            gpu.profile.pcie_bandwidth_gbps * 1e9 * (util / 100.0) * comm_mult * 0.2
                        )
                            gpu.pcie_tx_bytes_per_sec = base_tx * self._rng.uniform(0.9, 1.1)
                            gpu.pcie_rx_bytes_per_sec = base_tx * self._rng.uniform(0.9, 1.1)
                    else:
                        gpu.nvlink_tx_bytes_per_sec = 0.0
                        gpu.nvlink_rx_bytes_per_sec = 0.0

    def _update_thermal(self, cluster: SimCluster) -> None:
        """Thermal evolution with inertia, rack density, hotspots, cooling regimes.

        Replaces the instantaneous EMA with a lumped-capacitance ODE per GPU
        (T_{t+1} = T_t + a·P − b·(T−T_amb) + ε), rack-level kW density regimes,
        persistent hotspots that recover gradually, cooling-regime-dependent
        recovery, and CONTINUOUS thermal/power slowdown (not a binary flag).
        See aurelius/simulation/cluster/thermal.py.
        """
        cfg = self._serving_config or None
        for region in cluster.regions.values():
            # Aggregate power per rack_id (a rack holds multiple nodes) so density
            # reflects the whole rack, not a single node.
            rack_power_w: dict[str, float] = {}
            for node in region.nodes:
                rack_power_w[node.rack_id] = rack_power_w.get(node.rack_id, 0.0) + sum(
                    g.power_watts for g in node.gpus
                )

            for node in region.nodes:
                rt = node.rack_thermal
                regime = therm.resolve_cooling_regime(
                    node.cooling_regime if rt is None else rt.cooling_regime
                )

                # Rack power density (kW, whole rack) → density regime.
                job_powers = [g.power_watts for g in node.gpus]
                rack_kw_jobs = rack_power_w.get(node.rack_id, sum(job_powers)) / 1000.0
                density_regime = therm.rack_density_regime(rack_kw_jobs, regime, cfg)
                rack_kw = therm.rack_heat_kw(
                    [rack_power_w.get(node.rack_id, sum(job_powers))], density_regime, cfg
                )

                # Sustained power fraction (mean board power vs cap).
                caps = [max(1.0, g.power_cap_watts) for g in node.gpus]
                sustained = (
                    sum(g.power_watts for g in node.gpus) / sum(caps) if caps else 0.0
                )

                # Hotspot risk + persistence (hotspots linger, recover slowly).
                airflow_quality = rt.airflow.quality if rt is not None else 1.0
                risk = therm.hotspot_risk(rack_kw, airflow_quality, sustained, regime, cfg)
                prev_hot = rt.hotspot.severity if rt is not None else 0.0
                hot = therm.hotspot_step(prev_hot, risk, cfg)

                # Local inlet temperature: ambient + recirculation + variance +
                # any event-driven heat (a sustained cooling fault persists until
                # its _end event; GPU recovery lag comes from thermal inertia).
                base_ambient = region.ambient_temp_c + node.event_heat_c
                inlet = therm.inlet_temperature(base_ambient, hot, regime, self._rng, cfg)

                # Back-compat rack-heat proxy (kept for any legacy readers).
                node.rack_heat_delta_c = max(0.0, inlet - region.ambient_temp_c)

                peak_temp = 0.0
                for gpu in node.gpus:
                    pclass = therm.power_class_for_model(gpu.profile.model_name)
                    # Thermal inertia ODE step toward the local inlet.
                    t_next = therm.temperature_step(
                        gpu.temperature_c, gpu.power_watts, inlet, pclass, regime,
                        self._rng, cfg,
                    )
                    gpu.temperature_c = t_next
                    if gpu.thermal is not None:
                        gpu.thermal.inertia.temp_c = t_next
                        gpu.thermal.inertia.last_power_w = gpu.power_watts

                    # Continuous thermal + power slowdown.
                    s_thermal = therm.thermal_slowdown_frac(t_next, pclass, cfg)
                    s_power = therm.power_slowdown_frac(
                        gpu.power_watts, gpu.power_cap_watts, cfg
                    )
                    gpu.thermal_slowdown_frac = s_thermal
                    gpu.power_slowdown_frac = s_power
                    # Derived bool kept for back-compat (connector/metrics).
                    gpu.thermal_throttle_active = s_thermal > 0.0
                    if gpu.thermal is not None:
                        gpu.thermal.thermal_throttle.slowdown_frac = s_thermal
                        gpu.thermal.power_throttle.slowdown_frac = s_power
                        if s_thermal > 0.0:
                            gpu.thermal.thermal_throttle.throttle_events += 1
                    peak_temp = max(peak_temp, t_next)

                # Persist rack thermal state + violations/telemetry.
                if rt is not None:
                    rt.density.rack_kw = rack_kw
                    rt.density.regime = density_regime
                    rt.hotspot.severity = hot
                    rt.hotspot.risk = risk
                    rt.hotspot.persisted_ticks = (
                        rt.hotspot.persisted_ticks + 1 if hot > 0.3 else 0
                    )
                    rt.ambient.ambient_c = base_ambient
                    rt.ambient.inlet_c = inlet
                    rt.peak_gpu_temp_c = peak_temp
                    # Airflow degrades under sustained critical density, recovers slowly.
                    if density_regime == therm.RackDensityRegime.CRITICAL:
                        rt.airflow.quality = max(0.3, rt.airflow.quality - 0.1)
                        rt.airflow.instability = min(1.0, rt.airflow.instability + 0.15)
                    else:
                        rt.airflow.quality = min(1.0, rt.airflow.quality + 0.05)
                        rt.airflow.instability = max(0.0, rt.airflow.instability - 0.1)
                    rt.zone.zone_utilization = min(1.0, rack_kw / max(
                        1.0, thermal_value("rack_density_critical_kw", cfg) * regime.density_mult
                    ))
                    # Thermal excursion: any GPU past its throttle onset.
                    excursion = any(
                        g.thermal_slowdown_frac > 0.0 for g in node.gpus
                    )
                    rt.violation.last_tick_excursion = excursion
                    if excursion:
                        rt.violation.excursions += 1
                    if density_regime == therm.RackDensityRegime.CRITICAL:
                        rt.violation.cooling_alarms += 1
                    rt.migration_risk.risk = max(risk, min(1.0, hot))

    def _update_queues(self, cluster: SimCluster) -> None:
        """Update queue state with the inference-serving realism layer.

        Erlang-C/M/M/c baseline + convex saturation amplification + exploding
        latency tails + decomposed TTFT/TPOT + a batching/replica tradeoff.
        See aurelius/simulation/cluster/serving.py and calibration.py.
        """
        tick = cluster.tick
        hour = (tick - 1) % 24  # diurnal modulation (peak ~14:00)
        cfg = getattr(self, "_serving_config", None)

        for region in cluster.regions.values():
            for queue in region.queues:
                diurnal_factor = max(
                    0.1, 1.0 + queue.diurnal_amplitude * math.sin(math.pi * (hour - 6) / 12)
                )
                arrival_rate = queue.base_arrival_rate_per_sec * diurnal_factor

                # Bursty arrivals (Markov-modulated, seedable) on top of diurnal.
                # OFF by default — opt-in per scenario via serving_config.enable_bursts
                # so the canonical detection scenarios keep deterministic arrivals.
                if cfg and cfg.get("enable_bursts"):
                    queue.in_burst = serving.step_burst_state(queue.in_burst, self._rng, cfg)
                    arrival_rate *= serving.arrival_multiplier(queue.in_burst, cfg)
                if queue.surge_active:
                    arrival_rate *= queue.surge_multiplier
                queue.arrival_rate_per_sec = arrival_rate

                workload = self._find_workload_for_service(
                    queue.service_id, region.region_id, cluster
                )
                if workload is None:
                    queue.service_rate_per_sec = 0.01
                    queue.queue_depth = min(
                        queue.queue_depth + int(arrival_rate * 3600 * 0.1), 10000
                    )
                    queue.queue_wait_p95_ms = 60000.0
                    continue

                gpu_util = self._workload_effective_util(workload, cluster)
                warmup_factor = max(
                    0.2,
                    1.0 - workload.cold_start_warmup_ticks_remaining / _COLD_START_WARMUP_TICKS,
                )
                profile = self._get_workload_gpu_profile(workload, cluster)
                base_tps_per_gpu = (
                    profile.tokens_per_sec_at_full_util if profile is not None else 1000.0
                )
                tokens_per_sec = base_tps_per_gpu * (gpu_util / 100.0) * warmup_factor

                # Continuous thermal + power slowdown (NOT a binary throttle):
                # throughput = base · (1 − s_thermal − s_power), using the worst
                # GPU in the workload. Sustained heat materially cuts throughput.
                wl_gpus = self._workload_gpus(workload, cluster)
                s_thermal = max((g.thermal_slowdown_frac for g in wl_gpus), default=0.0)
                s_power = max((g.power_slowdown_frac for g in wl_gpus), default=0.0)
                thermal_tput = therm.throughput_factor(s_thermal, s_power)
                tokens_per_sec *= thermal_tput

                # Topology penalty for multi-GPU comm-heavy workloads.
                topo_penalty = 1.0 - (1.0 - workload.topology_score) * {
                    "low": 0.1, "medium": 0.3, "high": 0.5,
                }.get(workload.communication_intensity, 0.1)
                tokens_per_sec *= topo_penalty

                replicas = max(1, len(workload.gpu_ids))

                # Active sequences (offered concurrency) drive batching + contention.
                active_seqs = min(int(arrival_rate * 0.5) + queue.queue_depth // 100, 1024)

                # Batching/replica tradeoff: spreading the same load over more
                # replicas pushes each below the batching knee → lower tput/GPU.
                # Cache-aware: shared prefixes pack batches better; KV pressure
                # thins them.
                base_eff = serving.batching_efficiency(active_seqs, replicas, cfg)
                cache = workload.cache
                kv_pressure = cache.pressure.pressure if cache else 0.0
                hit_rate = cache.prefix.hit_rate if cache else 0.0
                batch_eff = kvc.cache_aware_batch_efficiency(
                    base_eff, hit_rate, kv_pressure, cfg
                )
                # Reroute churn fragments decode cohorts → η_batch degrades.
                migstate = workload.migration
                churn = migstate.route_shift.churn_rate if migstate else 0.0
                if migstate is not None:
                    batch_eff = mig.batch_efficiency_under_churn(batch_eff, churn, cfg)
                    migstate.cohort.efficiency = batch_eff
                tokens_per_sec *= batch_eff

                total_tokens_per_sec = tokens_per_sec * replicas
                service_rate = (
                    total_tokens_per_sec / _TOKENS_PER_REQUEST if _TOKENS_PER_REQUEST > 0 else 0.0
                )
                queue.service_rate_per_sec = max(0.01, service_rate)
                workload.effective_tokens_per_second = total_tokens_per_sec
                workload.effective_requests_per_second = service_rate

                rho = min(0.999, arrival_rate / queue.service_rate_per_sec)

                # Discrete-time backlog update (accelerates in overload).
                net_arrival = arrival_rate - queue.service_rate_per_sec
                delta = int(net_arrival * 3600 * cluster.tick_duration_hours * 0.01)
                queue.queue_depth = max(0, min(queue.queue_depth + delta, 50000))

                # Queue wait: Erlang-C mean × convex saturation amplifier.
                mu_per = queue.service_rate_per_sec / replicas
                mean_wait_s = serving.erlang_c_wait_s(arrival_rate, mu_per, replicas)
                if not math.isfinite(mean_wait_s):
                    mean_wait_s = 60.0
                mean_wait_s = min(60.0, mean_wait_s * serving.saturation_amplifier(rho, cfg))
                # Proxy/ingress bottleneck amplifies queue wait from OFFERED load
                # (arrivals), independent of replica count — replica count alone
                # does NOT set throughput once the proxy saturates.
                if migstate is not None:
                    proxy_sat = mig.proxy_saturation_factor(arrival_rate, replicas, cfg)
                    migstate.proxy.saturation_factor = proxy_sat
                else:
                    proxy_sat = 1.0
                mean_wait_s = min(60.0, mean_wait_s * proxy_sat)
                mean_wait_ms = mean_wait_s * 1000.0

                p95_mult, p99_mult = serving.tail_multipliers(rho, cfg)
                queue.queue_wait_p95_ms = mean_wait_ms * (p95_mult / 2.0 + 1.0)

                # Contention is PER-REPLICA: more replicas means each handles
                # fewer concurrent sequences → less scheduler/decode contention.
                # (This is the autoscaling benefit; it trades off against the
                # batching-efficiency loss applied to throughput above.)
                active_per_replica = active_seqs / replicas

                # Decomposed TTFT = queue + prefill + active-seq contention + KV
                # stall + cold-route penalty + recompute penalty. Prefill shrinks
                # under prefix reuse; the contention/KV part is amplified by KV
                # pressure; cold reroutes and preemption add explicit penalties.
                prompt_tokens = _TOKENS_PER_REQUEST  # representative; heavy-tailed dist = remaining gap
                kv = min(1.0, kv_pressure)
                savings = cache.prefix.prefill_savings_frac if cache else 0.0
                eff_prompt = prompt_tokens * (1.0 - savings)   # prefix reuse cuts prefill
                # Compute part only (queue wait passed as 0), then amplify by KV
                # pressure and add cache penalties; finally add the queue wait.
                ttft_compute = serving.ttft_ms(0.0, eff_prompt, active_per_replica, kv, 1.0, cfg)
                ttft_compute *= kvc.pressure_ttft_multiplier(kv, cfg)
                if cache is not None:
                    ttft_compute += cache.affinity.cold_route_penalty_ms
                    ttft_compute += cache.preemption.recompute_penalty_ms
                # Migration startup penalty (drain/cold-start/requeue) during the
                # warmup window — cold starts materially hurt TTFT.
                if migstate is not None:
                    ttft_compute += migstate.warmup.startup_penalty_ms
                ttft_p50 = mean_wait_ms + ttft_compute
                # Migration amplifies the TAIL (p95/p99), not just the median.
                tail_mult = migstate.tail.uplift_mult if migstate else 1.0
                queue.ttft_p50_ms = ttft_p50
                queue.ttft_p95_ms = ttft_p50 * p95_mult * tail_mult
                queue.ttft_p99_ms = ttft_p50 * p99_mult * tail_mult

                # Decomposed TPOT: base × throttle + per-replica decode contention.
                # Throttle factor scales with the CONTINUOUS thermal+power slowdown
                # (slower clocks → higher inter-token latency), not a binary flag.
                throttle_factor = 1.0 / max(0.05, therm.throughput_factor(s_thermal, s_power))
                tpot_p50 = serving.tpot_ms(_BASE_TPOT_MS, active_per_replica, throttle_factor, cfg)
                queue.tpot_p50_ms = tpot_p50
                queue.tpot_p95_ms = tpot_p50 * 2.0
                queue.tpot_p99_ms = tpot_p50 * 4.0

                avg_output_tokens = 128
                queue.latency_p50_ms = ttft_p50 + tpot_p50 * avg_output_tokens
                queue.latency_p95_ms = queue.ttft_p95_ms + queue.tpot_p95_ms * avg_output_tokens
                queue.latency_p99_ms = queue.ttft_p99_ms + queue.tpot_p99_ms * avg_output_tokens

                sla_ms = workload.latency_sla_p99_ms or _SLA_P99_DEFAULT_MS
                if queue.latency_p99_ms > sla_ms:
                    timeout_rate = min(50.0, (queue.latency_p99_ms - sla_ms) / sla_ms * 10.0)
                    cluster.sla_violations += 1
                else:
                    timeout_rate = 0.0
                queue.timeout_rate_pct = timeout_rate

                queue.active_sequences = active_seqs
                queue.batch_size = min(active_seqs // replicas, 256)
                queue.tokens_per_second = total_tokens_per_sec
                queue.requests_per_second = service_rate
                queue.proxy_saturation = proxy_sat
                queue.batch_efficiency = batch_eff

                # Overload event: sustained queue pressure beyond the governor's
                # threshold is an operational incident worth counting.
                if migstate is not None and queue.queue_depth >= migration_value(
                    "governor_queue_pressure_qdepth", cfg
                ):
                    migstate.migration.overload_events += 1

                # Remember offered concurrency so the NEXT tick's KV pressure is
                # computed from it (deterministic pre/post ordering).
                if cache is not None:
                    cache.active_seqs_prev = float(active_seqs)

    def _update_kv_cache(self, cluster: SimCluster) -> None:
        """Realistic KV-cache / prefix-affinity / memory-pressure update.

        Runs BEFORE _update_queues so this tick's queue physics see the pressure,
        prefix hit rate, and pending cold-route/recompute penalties. Pressure is
        computed from last tick's offered concurrency (cache.active_seqs_prev),
        keeping the pre/post update order deterministic. See kv_cache.py.
        """
        cfg = self._serving_config or None
        for region in cluster.regions.values():
            for queue in region.queues:
                wl = self._find_workload_for_service(
                    queue.service_id, region.region_id, cluster
                )
                if wl is None or wl.cache is None:
                    continue
                cache = wl.cache
                profile = kvc.resolve_kv_profile(wl.model_kv_profile, cfg)

                # --- KV memory + pressure (scaling law) -----------------------
                batch = max(1.0, cache.active_seqs_prev)
                seq_len = max(1.0, float(wl.avg_seq_len_tokens))
                gpus = self._workload_gpus(wl, cluster)
                replicas = max(1, len(gpus))
                per_gpu_total = (
                    gpus[0].profile.memory_total_bytes if gpus else 80 * 1024**3
                )
                free_after_weights = max(1.0, per_gpu_total - wl.memory_required_bytes)
                budget = free_after_weights * kv_value("kv_reserved_budget_frac", cfg) * replicas

                allocated = kvc.kv_bytes(profile, batch, seq_len)
                frag = kvc.fragmentation_frac(batch, seq_len, profile, cfg)
                allocated_with_slack = (
                    allocated / (1.0 - frag) if frag < 0.999 else allocated
                )
                pressure = kvc.kv_pressure(allocated_with_slack, budget)
                if wl.kv_pressure_override is not None:
                    pressure = max(pressure, wl.kv_pressure_override)
                region_name = kvc.pressure_region(pressure, cfg)

                cache.kv.allocated_bytes = allocated_with_slack
                cache.kv.reserved_budget_bytes = budget
                cache.kv.batch_size = batch
                cache.kv.avg_seq_len = seq_len
                cache.kv.occupancy_frac = min(
                    1.0, allocated_with_slack / max(1.0, per_gpu_total * replicas)
                )
                cache.pressure.pressure = pressure
                cache.pressure.region = region_name
                cache.fragmentation.slack_frac = frag
                cache.fragmentation.slack_bytes = max(0.0, allocated_with_slack - allocated)

                # --- Locality confidence (reuse-driven warmup / decay) --------
                in_cold_window = cache.affinity.cold_warmup_ticks_remaining > 0
                reused = not in_cold_window
                cache.locality.confidence = kvc.locality_confidence_step(
                    cache.locality.confidence, reused, cfg
                )
                # High pressure thrashes the cache (LRU eviction) → confidence dips.
                if region_name in (kvc.PressureRegion.THROTTLING, kvc.PressureRegion.PREEMPTION):
                    cache.locality.confidence *= 1.0 - kv_value(
                        "locality_confidence_decay", cfg
                    )
                cache.warmup.warm = cache.locality.confidence > 0.7
                cache.warmup.ticks_warm = (
                    cache.warmup.ticks_warm + 1 if reused and cache.warmup.warm else 0
                )

                # --- Prefix hit rate (overlap × locality), honoring override --
                cache.prefix.overlap = wl.prefix_overlap
                cache.prefix.shared_prefix_tokens = wl.prefix_overlap * wl.avg_seq_len_tokens
                hit = kvc.prefix_hit_rate(wl.prefix_overlap, cache.locality.confidence, cfg)
                if wl.prefix_hit_override is not None:
                    hit = min(hit, wl.prefix_hit_override)
                cache.prefix.hit_rate = hit
                cache.prefix.prefill_savings_frac = kvc.prefill_savings_frac(hit, cfg)

                # --- Preemption / recompute / eviction under exhaustion -------
                seq_bytes = max(1.0, kvc.kv_bytes(profile, 1.0, seq_len))
                overflow_seqs = int(max(0.0, allocated_with_slack - budget) / seq_bytes)
                prob = kvc.preemption_probability(pressure, cfg)
                preempted = overflow_seqs
                if prob > 0.0 and self._rng.random() < prob:
                    preempted = max(preempted, 1)
                if preempted > 0:
                    cache.preemption.last_tick_count = preempted
                    cache.preemption.cumulative_count += preempted
                    cache.preemption.recompute_penalty_ms = kvc.recompute_penalty_ms(
                        preempted, seq_len, cfg
                    )
                    cache.eviction.last_tick_evictions = preempted
                    cache.eviction.cumulative_evictions += preempted
                else:
                    cache.preemption.last_tick_count = 0
                    cache.eviction.last_tick_evictions = 0
                    # Pending recompute penalty drains over subsequent ticks.
                    cache.preemption.recompute_penalty_ms *= 0.5

                # --- Cold-route penalty decay over the warmup window ----------
                if cache.affinity.cold_warmup_ticks_remaining > 0:
                    cache.affinity.cold_warmup_ticks_remaining -= 1
                    cache.affinity.cold_route_penalty_ms *= 0.5
                else:
                    cache.affinity.cold_route_penalty_ms = 0.0

                # --- Routing affinity score -----------------------------------
                on_home = wl.region_id == cache.routing.home_region
                cache.routing.affinity_score = (
                    cache.locality.confidence if on_home else cache.locality.confidence * 0.3
                )

                # --- Back-compat fracs + telemetry-facing queue fields --------
                wl.kv_cache_usage_frac = min(0.98, pressure)
                wl.prefix_cache_hit_rate_frac = hit
                tier = cache.routing.telemetry_tier
                # Missing/low telemetry hides KV internals (but NOT 'no pressure').
                if tier == "low":
                    queue.kv_cache_usage_pct = None
                    queue.prefix_cache_hit_rate_pct = None
                    queue.kv_pressure = None
                    queue.kv_pressure_region = None
                    queue.preemptions_total = None
                    queue.cache_fragmentation_frac = None
                else:
                    # kv_cache_usage mirrors vLLM gpu_cache_usage_perc — KV-block
                    # utilization (≈ pressure clamped to 1), NOT total-GPU-mem.
                    queue.kv_cache_usage_pct = min(1.0, pressure) * 100.0
                    queue.prefix_cache_hit_rate_pct = hit * 100.0
                    queue.kv_pressure = pressure
                    queue.kv_pressure_region = region_name
                    queue.preemptions_total = float(cache.preemption.cumulative_count)
                    queue.cache_fragmentation_frac = frag

    def _update_migration(self, cluster: SimCluster) -> None:
        """Advance migration/drain/cold-start state one tick (decay + bookkeeping).

        Runs after _update_kv_cache and before _update_queues so the queue
        physics see the current warmup penalty, batching cohort efficiency, and
        tail uplift. Penalties/instability decay over their windows so a single
        migration is a transient spike, not a permanent tax.
        """
        cfg = self._serving_config or None
        for wl in cluster.workloads.values():
            m = wl.migration
            if m is None:
                continue

            # Route churn decays toward 0 (recent-reroute intensity).
            m.route_shift.churn_rate *= 0.6

            # Warmup window counts down; the startup TTFT penalty drains with it.
            if m.warmup.ticks_remaining > 0:
                m.warmup.ticks_remaining -= 1
                m.warmup.startup_penalty_ms *= 0.5
                m.warmup.warm = m.warmup.ticks_remaining <= 0
            else:
                m.warmup.startup_penalty_ms = 0.0
                m.warmup.warm = True

            # Tail uplift relaxes back toward 1.0 once churn/instability subside.
            target_tail = mig.tail_uplift(
                m.rollout.instability, 0.0, m.route_shift.churn_rate,
                wl.cache.prefix.hit_rate if wl.cache else 0.0, cfg,
            ) if (m.route_shift.churn_rate > 0.01 or m.rollout.instability > 0.01) else 1.0
            m.tail.uplift_mult = max(1.0, 0.5 * m.tail.uplift_mult + 0.5 * target_tail)

            # Rollout hold window counts down; instability relaxes.
            if m.rollout.hold_ticks_remaining > 0:
                m.rollout.hold_ticks_remaining -= 1
            m.rollout.instability *= 0.7

            # PDB availability tracks current replica count vs the floor.
            m.pdb.available = max(0, len(wl.gpu_ids) - m.pdb.min_available)

            # Warm-pool occupancy: fraction of warm replicas currently serving.
            if m.warm_pool.size > 0:
                m.warm_pool.occupancy = min(1.0, len(wl.gpu_ids) / m.warm_pool.size)
            # Proxy saturation is computed in _update_queues from offered load.

    def _update_cost_accounting(self, cluster: SimCluster) -> None:
        """Accumulate cost and energy metrics for this tick."""
        tick_energy_kwh = 0.0
        tick_cost = 0.0
        tick_tokens = 0

        for region in cluster.regions.values():
            price_per_kwh = region.current_energy_price / 1000.0   # $/MWh → $/kWh
            for node in region.nodes:
                for gpu in node.gpus:
                    gpu_kwh = gpu.power_watts / 1000.0 * cluster.tick_duration_hours
                    tick_energy_kwh += gpu_kwh
                    tick_cost += gpu_kwh * price_per_kwh

            for queue in region.queues:
                tick_tokens += int(queue.tokens_per_second * 3600 * cluster.tick_duration_hours)

        cluster.total_energy_kwh += tick_energy_kwh
        cluster.total_energy_cost += tick_cost
        cluster.total_tokens_served += tick_tokens

    def _compute_tick_metrics(self, cluster: SimCluster, ts: datetime) -> TickMetrics:
        """Compute aggregated metrics for this tick."""
        tick_tokens = 0
        tick_energy_kwh = 0.0
        tick_cost = 0.0
        util_values: list[float] = []
        p99_values: list[float] = []
        p95_wait_values: list[float] = []
        ttft_p50_values: list[float] = []
        ttft_p95_values: list[float] = []
        ttft_p99_values: list[float] = []
        throttle_count = 0

        for region in cluster.regions.values():
            price_per_kwh = region.current_energy_price / 1000.0
            for node in region.nodes:
                for gpu in node.gpus:
                    util_values.append(gpu.utilization_pct)
                    if gpu.thermal_throttle_active:
                        throttle_count += 1
                    gpu_kwh = gpu.power_watts / 1000.0 * cluster.tick_duration_hours
                    tick_energy_kwh += gpu_kwh
                    tick_cost += gpu_kwh * price_per_kwh

            for queue in region.queues:
                tick_tokens += int(queue.tokens_per_second * 3600 * cluster.tick_duration_hours)
                if queue.latency_p99_ms is not None:
                    p99_values.append(queue.latency_p99_ms)
                if queue.queue_wait_p95_ms is not None:
                    p95_wait_values.append(queue.queue_wait_p95_ms)
                if queue.ttft_p50_ms is not None:
                    ttft_p50_values.append(queue.ttft_p50_ms)
                if queue.ttft_p95_ms is not None:
                    ttft_p95_values.append(queue.ttft_p95_ms)
                if queue.ttft_p99_ms is not None:
                    ttft_p99_values.append(queue.ttft_p99_ms)

        # Cache / locality KPIs aggregated across workloads.
        kv_pressures: list[float] = []
        hit_rates: list[float] = []
        loc_confs: list[float] = []
        frag_fracs: list[float] = []
        affinity_scores: list[float] = []
        preemption_count = 0
        recompute_count = 0
        cold_reroute_count = 0
        eviction_count = 0
        for wl in cluster.workloads.values():
            c = wl.cache
            if c is None:
                continue
            kv_pressures.append(c.pressure.pressure)
            hit_rates.append(c.prefix.hit_rate)
            loc_confs.append(c.locality.confidence)
            frag_fracs.append(c.fragmentation.slack_frac)
            affinity_scores.append(c.routing.affinity_score)
            preemption_count += c.preemption.last_tick_count
            recompute_count += 1 if c.preemption.recompute_penalty_ms > 0 else 0
            cold_reroute_count += c.affinity.cold_reroute_count
            eviction_count += c.eviction.last_tick_evictions

        # Migration / drain / cold-start KPIs aggregated across workloads.
        reroute_count = 0
        veto_count = 0
        drain_total = 0.0
        startup_latencies: list[float] = []
        warmup_active = 0
        batch_effs: list[float] = []
        churns: list[float] = []
        proxy_sats: list[float] = []
        cold_starts = 0
        warm_occ: list[float] = []
        rollback_count = 0
        overload_events = 0
        for wl in cluster.workloads.values():
            m = wl.migration
            if m is None:
                continue
            reroute_count += m.route_shift.reroute_count
            veto_count += m.migration.veto_count
            drain_total += m.drain.drain_seconds_total
            startup_latencies.append(m.startup.last_cold_seconds)
            if m.warmup.ticks_remaining > 0:
                warmup_active += 1
            batch_effs.append(m.cohort.efficiency)
            churns.append(m.route_shift.churn_rate)
            proxy_sats.append(m.proxy.saturation_factor)
            cold_starts += m.coldstart.cold_start_count
            if m.warm_pool.size > 0:
                warm_occ.append(m.warm_pool.occupancy)
            rollback_count += m.rollout.rollback_count
            overload_events += m.migration.overload_events
            if m.migration.last_veto_reason == "thermal_hot_destination":
                pass  # veto reasons aggregated below via thermal_vetoes

        # Thermal / cooling / power KPIs aggregated across GPUs + racks.
        gpu_temps: list[float] = []
        s_thermals: list[float] = []
        s_powers: list[float] = []
        throttle_events = 0
        inlet_temps: list[float] = []
        hotspots: list[float] = []
        rack_kws: list[float] = []
        thermal_excursions = 0
        cooling_alarms = 0
        thermal_vetoes = 0
        for region in cluster.regions.values():
            for node in region.nodes:
                for gpu in node.gpus:
                    gpu_temps.append(gpu.temperature_c)
                    s_thermals.append(gpu.thermal_slowdown_frac)
                    s_powers.append(gpu.power_slowdown_frac)
                    if gpu.thermal_slowdown_frac > 0.0:
                        throttle_events += 1
                rt = node.rack_thermal
                if rt is not None:
                    inlet_temps.append(rt.ambient.inlet_c)
                    hotspots.append(rt.hotspot.severity)
                    rack_kws.append(rt.density.rack_kw)
                    thermal_excursions += rt.violation.excursions
                    cooling_alarms += rt.violation.cooling_alarms
                    thermal_vetoes += rt.migration_risk.veto_count
        for wl in cluster.workloads.values():
            m = wl.migration
            if m is not None and m.migration.last_veto_reason == "thermal_hot_destination":
                thermal_vetoes += 1

        mean_util = sum(util_values) / len(util_values) if util_values else 0.0
        p99_lat = max(p99_values) if p99_values else None
        p95_wait = max(p95_wait_values) if p95_wait_values else None

        cost_per_token = tick_cost / tick_tokens if tick_tokens > 0 else None
        energy_joules = tick_energy_kwh * 3_600_000
        tokens_per_joule = tick_tokens / energy_joules if energy_joules > 0 else None

        topo_scores = [wl.topology_score for wl in cluster.workloads.values()]
        mean_topo = sum(topo_scores) / len(topo_scores) if topo_scores else 1.0

        return TickMetrics(
            tick=cluster.tick,
            timestamp=ts,
            total_energy_cost=tick_cost,
            total_tokens=tick_tokens,
            total_energy_kwh=tick_energy_kwh,
            cost_per_token=cost_per_token,
            tokens_per_joule=tokens_per_joule,
            mean_gpu_util_pct=mean_util,
            p95_latency_ms=(
                max(p99_values[:-1]) if len(p99_values) > 1
                else (p99_values[0] * 0.6 if p99_values else None)
            ),
            p99_latency_ms=p99_lat,
            queue_wait_p95_ms=p95_wait,
            sla_violations=cluster.sla_violations,
            thermal_throttle_gpu_count=throttle_count,
            migration_count=cluster.migration_count,
            mean_topology_score=mean_topo,
            kv_pressure_max=max(kv_pressures) if kv_pressures else None,
            prefix_hit_rate_mean=(
                sum(hit_rates) / len(hit_rates) if hit_rates else None
            ),
            preemption_count=preemption_count,
            recompute_count=recompute_count,
            cold_reroute_count=cold_reroute_count,
            cache_eviction_count=eviction_count,
            locality_confidence_mean=(
                sum(loc_confs) / len(loc_confs) if loc_confs else None
            ),
            cache_fragmentation_frac_mean=(
                sum(frag_fracs) / len(frag_fracs) if frag_fracs else None
            ),
            routing_affinity_score_mean=(
                sum(affinity_scores) / len(affinity_scores) if affinity_scores else None
            ),
            ttft_p50_ms=max(ttft_p50_values) if ttft_p50_values else None,
            ttft_p95_ms=max(ttft_p95_values) if ttft_p95_values else None,
            ttft_p99_ms=max(ttft_p99_values) if ttft_p99_values else None,
            reroute_count=reroute_count,
            migration_veto_count=veto_count,
            drain_seconds_total=drain_total,
            startup_latency_s_max=max(startup_latencies) if startup_latencies else None,
            warmup_active_count=warmup_active,
            batch_efficiency_mean=sum(batch_effs) / len(batch_effs) if batch_effs else None,
            route_churn_mean=sum(churns) / len(churns) if churns else None,
            proxy_saturation_max=max(proxy_sats) if proxy_sats else None,
            cold_start_count=cold_starts,
            warm_pool_occupancy_mean=sum(warm_occ) / len(warm_occ) if warm_occ else None,
            rollback_count=rollback_count,
            overload_events=overload_events,
            max_gpu_temp_c=max(gpu_temps) if gpu_temps else None,
            max_rack_inlet_c=max(inlet_temps) if inlet_temps else None,
            thermal_slowdown_pct_mean=(
                100.0 * sum(s_thermals) / len(s_thermals) if s_thermals else None
            ),
            power_slowdown_pct_mean=(
                100.0 * sum(s_powers) / len(s_powers) if s_powers else None
            ),
            thermal_throttle_events=throttle_events,
            hotspot_severity_max=max(hotspots) if hotspots else None,
            rack_density_kw_max=max(rack_kws) if rack_kws else None,
            thermal_excursions=thermal_excursions,
            cooling_alarms=cooling_alarms,
            thermal_migration_vetoes=thermal_vetoes,
        )

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _derive_region_interconnect_class(self, region: "SimRegion") -> str:  # type: ignore[name-defined]
        """Derive the worst-case interconnect_class for a region from node topology_class.

        nvswitch → nvlink_full; nvlink4/nvlink2 → nvlink_partial;
        pcie_multi_numa → cross_numa; pcie → pcie; mixed/poor → pcie.
        If any node has a poor class (pcie/cross_numa), the region is poor.
        """
        _CLASS_RANK = {
            "nvswitch": 4,
            "nvlink4": 3,
            "nvlink2": 3,
            "pcie_multi_numa": 2,
            "pcie": 1,
        }
        _CLASS_TO_INTERCONNECT = {
            "nvswitch": "nvlink_full",
            "nvlink4": "nvlink_partial",
            "nvlink2": "nvlink_partial",
            "pcie_multi_numa": "cross_numa",
            "pcie": "pcie",
        }
        worst_rank = 99
        worst_class = "unknown"
        for node in region.nodes:
            cls = node.labels.get("topology-class", "unknown")
            rank = _CLASS_RANK.get(cls, 0)
            if rank < worst_rank:
                worst_rank = rank
                worst_class = cls
        return _CLASS_TO_INTERCONNECT.get(worst_class, "unknown")

    def _find_workload_for_service(
        self, service_id: str, region_id: str, cluster: SimCluster
    ) -> Optional[SimWorkload]:
        for wl in cluster.workloads.values():
            if wl.service_id == service_id and wl.region_id == region_id:
                return wl
        return None

    def _workload_gpus(self, workload: SimWorkload, cluster: SimCluster) -> list[SimGPU]:
        gpus = []
        for region in cluster.regions.values():
            for node in region.nodes:
                for gpu in node.gpus:
                    if gpu.gpu_id in workload.gpu_ids:
                        gpus.append(gpu)
        return gpus

    def _workload_effective_util(self, workload: SimWorkload, cluster: SimCluster) -> float:
        gpus = self._workload_gpus(workload, cluster)
        if not gpus:
            return 0.0
        return sum(g.effective_utilization_pct for g in gpus) / len(gpus)

    def _get_workload_gpu_profile(
        self, workload: SimWorkload, cluster: SimCluster
    ) -> Optional[GPUProfile]:
        gpus = self._workload_gpus(workload, cluster)
        if gpus:
            return gpus[0].profile
        return None

    def _compute_topology_score(self, workload: SimWorkload, cluster: SimCluster) -> float:
        """Compute 0-1 topology score for a workload's current placement."""
        if workload.gpu_count_required <= 1:
            return 1.0
        if len(workload.gpu_ids) < 2:
            return 1.0

        # Find links between workload GPUs
        gpu_set = set(workload.gpu_ids)
        best_link_type = "SYS"   # worst default

        for region in cluster.regions.values():
            for node in region.nodes:
                for link in node.topology_links:
                    if link.gpu_a in gpu_set and link.gpu_b in gpu_set:
                        # Prefer higher bandwidth link types
                        if _link_rank(link.link_type) > _link_rank(best_link_type):
                            best_link_type = link.link_type

        score_map = {
            "NVSWITCH": 1.0,
            "NV4": 0.95,
            "NV2": 0.9,
            "PIX": 0.75,
            "PXB": 0.65,
            "PHB": 0.5,
            "NODE": 0.4,
            "SYS": 0.25,
            "RACK": 0.15,
            "REGION": 0.05,
        }
        base_score = score_map.get(best_link_type, 0.25)

        # Communication intensity multiplier: poor topology hurts high-comm more.
        comm_weight = {"low": 0.3, "medium": 0.6, "high": 1.0}.get(
            workload.communication_intensity, 0.3
        )
        # Penalty scales with both topology badness (1 - base_score) and comm
        # intensity: a perfect link (base_score=1.0) is never penalized, while a
        # poor link penalizes high-comm workloads far more than low-comm ones.
        return base_score * (1.0 - comm_weight * (1.0 - base_score))

    # ------------------------------------------------------------------
    # Connector data generators (fake connector payloads)
    # ------------------------------------------------------------------

    def get_cluster_state(self) -> ClusterState:
        """Convert mutable simulation state to canonical frozen ClusterState.

        Uses the actual field names from aurelius/state/models.py.
        """
        cluster = self._cluster
        ts = self._tick_timestamp(cluster.tick)
        prov = Provenance(
            source="simulator",
            fetched_at=ts,
            confidence="high",
            is_sandbox=True,
        )

        region_states: dict[str, RegionState] = {}

        for region in cluster.regions.values():
            node_states: dict[str, NodeState] = {}
            service_states: dict[str, InferenceServiceState] = {}

            for node in region.nodes:
                gpu_states: dict[str, GPUState] = {}
                allocated_count = 0

                for gpu in node.gpus:
                    throttle_bits = 8 if gpu.thermal_throttle_active else 0
                    gs = GPUState(
                        gpu_uuid=gpu.uuid,
                        node_id=gpu.node_id,
                        region=region.region_id,
                        timestamp=ts,
                        provenance=prov,
                        gpu_index=gpu.gpu_index,
                        gpu_type=gpu.profile.model_name,
                        util_pct=max(0.0, min(100.0, gpu.utilization_pct)),
                        sm_active_ratio=max(0.0, min(1.0, gpu.sm_activity_pct / 100.0)),
                        mem_used_mb=gpu.memory_used_bytes / (1024 * 1024),
                        mem_free_mb=gpu.memory_free_bytes / (1024 * 1024),
                        mem_total_mb=gpu.profile.memory_total_bytes / (1024 * 1024),
                        power_w=max(0.0, gpu.power_watts),
                        temp_c=max(0.0, min(_MAX_REALISTIC_TEMP_C, gpu.temperature_c)),
                        clocks_event_reasons=throttle_bits,
                        xid_last=gpu.xid_error_count if gpu.xid_error_count > 0 else None,
                        nvlink_tx_bytes_per_s=(
                            gpu.nvlink_tx_bytes_per_sec if gpu.nvlink_tx_bytes_per_sec > 0 else None
                        ),
                        nvlink_rx_bytes_per_s=(
                            gpu.nvlink_rx_bytes_per_sec if gpu.nvlink_rx_bytes_per_sec > 0 else None
                        ),
                        pcie_tx_bytes_per_s=(
                            gpu.pcie_tx_bytes_per_sec if gpu.pcie_tx_bytes_per_sec > 0 else None
                        ),
                        pcie_rx_bytes_per_s=(
                            gpu.pcie_rx_bytes_per_sec if gpu.pcie_rx_bytes_per_sec > 0 else None
                        ),
                        is_schedulable=gpu.assigned_workload_id is None,
                    )
                    gpu_states[gpu.uuid] = gs
                    if gpu.assigned_workload_id is not None:
                        allocated_count += 1

                ns = NodeState(
                    node_id=node.node_id,
                    region=region.region_id,
                    timestamp=ts,
                    provenance=prov,
                    zone=node.zone,
                    rack_id=node.rack_id,
                    instance_type=node.instance_type,
                    gpu_capacity=node.gpu_count,
                    gpu_allocatable=node.gpu_count,
                    gpu_allocated=allocated_count,
                    labels=dict(node.labels),
                    taints=list(node.taints),
                    schedulable=True,
                    gpus=gpu_states,
                )
                node_states[node.node_id] = ns

            # Build InferenceServiceState from queues in this region
            for queue in region.queues:
                workload = self._find_workload_for_service(
                    queue.service_id, region.region_id, cluster
                )
                runtime = (
                    "vllm"
                    if workload and workload.workload_type in ("inference", "embedding")
                    else "unknown"
                )
                node_id = workload.node_ids[0] if workload and workload.node_ids else None

                iss = InferenceServiceState(
                    service_id=queue.service_id,
                    engine=runtime,
                    timestamp=ts,
                    provenance=prov,
                    region=region.region_id,
                    node_id=node_id,
                    requests_running=max(0.0, float(queue.active_sequences)),
                    requests_waiting=max(0.0, float(queue.queue_depth)),
                    p50_latency_ms=queue.latency_p50_ms,
                    p95_latency_ms=queue.latency_p95_ms,
                    p99_latency_ms=queue.latency_p99_ms,
                    ttft_p50_ms=queue.ttft_p50_ms,
                    ttft_p95_ms=queue.ttft_p95_ms,
                    ttft_p99_ms=queue.ttft_p99_ms,
                    queue_time_p95_ms=queue.queue_wait_p95_ms,
                    kv_cache_usage=(
                        queue.kv_cache_usage_pct / 100.0
                        if queue.kv_cache_usage_pct is not None else None
                    ),
                    prefix_cache_hit_rate=(
                        queue.prefix_cache_hit_rate_pct / 100.0
                        if queue.prefix_cache_hit_rate_pct is not None else None
                    ),
                    preemptions_total=queue.preemptions_total,
                    tokens_per_s=max(0.0, queue.tokens_per_second),
                    error_rate_pct=max(0.0, min(100.0, queue.timeout_rate_pct)),
                )
                service_states[queue.service_id] = iss

            # Build EnergyState for this region
            power_draw_kw = sum(
                gpu.power_watts / 1000.0
                for node in region.nodes
                for gpu in node.gpus
            )
            energy = EnergyState(
                region=region.region_id,
                timestamp=ts,
                provenance=prov,
                price_per_mwh=region.current_energy_price,
                real_time_price_per_mwh=region.current_energy_price,
                carbon_gco2_per_kwh=region.current_carbon_intensity,
                power_draw_kw=power_draw_kw,
            )

            # Compute spare capacity
            total_gpus = sum(n.gpu_capacity or 0 for n in node_states.values())
            alloc_gpus = sum(n.gpu_allocated or 0 for n in node_states.values())
            spare_pct = ((total_gpus - alloc_gpus) / total_gpus * 100.0
                         if total_gpus > 0 else None)

            # Build a minimal TopologyState for the region based on node topology classes.
            # interconnect_class summarizes the worst-case link in the region so the
            # classifier can detect topology-bound pressure without live nvidia-smi data.
            region_interconnect = self._derive_region_interconnect_class(region)
            # Populate pair_levels from topology_links on each node so PlacementScorer works.
            pair_levels: dict[tuple[str, str], TopologyLinkType] = {}
            for node in region.nodes:
                for link in node.topology_links:
                    try:
                        lt = TopologyLinkType(link.link_type.lower())
                    except ValueError:
                        lt = TopologyLinkType.SYS
                    a, b = link.gpu_a, link.gpu_b
                    key: tuple[str, str] = (a, b) if a < b else (b, a)
                    pair_levels[key] = lt
            all_gpu_uuids = tuple(
                gpu.uuid for node in region.nodes for gpu in node.gpus
            )
            region_topology = TopologyState(
                node_id=f"{region.region_id}-aggregate",
                timestamp=ts,
                provenance=prov,
                gpu_uuids=all_gpu_uuids,
                numa_affinity={},
                pair_levels=pair_levels,
                interconnect_class=region_interconnect,
            )

            rs = RegionState(
                region=region.region_id,
                timestamp=ts,
                provenance=prov,
                nodes=node_states,
                services=service_states,
                energy=energy,
                spare_capacity_pct=spare_pct,
                topology=region_topology,
            )
            region_states[region.region_id] = rs

        return ClusterState(
            timestamp=ts,
            provenance=prov,
            regions=region_states,
            is_partial=False,
        )

    def get_dcgm_prometheus_text(self, node_id: str) -> str:
        """Generate Prometheus DCGM-format metrics text for a node."""
        from .fakes.prometheus_text import generate_dcgm_metrics_text
        cluster = self._cluster
        node = self._find_node(node_id, cluster)
        if node is None:
            return ""
        return generate_dcgm_metrics_text(node)

    def get_vllm_prometheus_text(self, service_id: str) -> str:
        """Generate vLLM Prometheus metrics text for a service."""
        from .fakes.prometheus_text import generate_vllm_metrics_text
        cluster = self._cluster
        for region in cluster.regions.values():
            for queue in region.queues:
                if queue.service_id == service_id:
                    workload = self._find_workload_for_service(
                        service_id, region.region_id, cluster
                    )
                    return generate_vllm_metrics_text(queue, workload)
        return ""

    def get_kubernetes_node_list(self) -> dict[str, Any]:
        """Generate fake V1NodeList API payload."""
        from .fakes.kubernetes_payloads import generate_node_list
        return generate_node_list(self._cluster)

    def get_kubernetes_pod_list(self) -> dict[str, Any]:
        """Generate fake V1PodList API payload."""
        from .fakes.kubernetes_payloads import generate_pod_list
        return generate_pod_list(self._cluster)

    def get_nvidia_smi_topo_text(self, node_id: str) -> str:
        """Generate fake nvidia-smi topo -m text output for a node."""
        from .fakes.topology_text import generate_topo_text
        cluster = self._cluster
        node = self._find_node(node_id, cluster)
        if node is None:
            return ""
        return generate_topo_text(node)

    def get_nvidia_smi_list_text(self, node_id: str) -> str:
        """Generate fake nvidia-smi -L text output for a node."""
        from .fakes.topology_text import generate_gpu_list_text
        cluster = self._cluster
        node = self._find_node(node_id, cluster)
        if node is None:
            return ""
        return generate_gpu_list_text(node)

    def _find_node(self, node_id: str, cluster: SimCluster) -> Optional[SimNode]:
        for region in cluster.regions.values():
            for node in region.nodes:
                if node.node_id == node_id:
                    return node
        return None

    # ------------------------------------------------------------------
    # Migration support (used by optimizer/tests)
    # ------------------------------------------------------------------

    def migrate_workload(
        self,
        workload_id: str,
        target_region_id: str,
        target_node_ids: Optional[list[str]] = None,
        *,
        respect_governor: bool = False,
    ) -> bool:
        """Simulate workload migration to another region (NOT free, NOT instant).

        This is the benchmark feedback mechanism for optimizer policies. Basic
        safety checks reject invalid migrations (unknown/ same region, capacity,
        migration_allowed=False). A PodDisruptionBudget that forbids eviction
        ALWAYS blocks the migration (drain stall). When ``respect_governor`` is
        set, the migration governor may additionally veto under queue pressure,
        strong cache affinity, rollout instability, incomplete warmup, or a
        startup-heavy / scale-from-zero path (do-nothing is often safest).

        On success the full migration cost is applied: Kubernetes-style drain +
        engine-specific heavy-tailed cold start + cache loss + batching
        disruption + p95/p99 tail uplift. See migration.py / migration_model.py.

        Returns True if migration was applied, False if blocked/vetoed.
        """
        cluster = self._cluster
        workload = cluster.workloads.get(workload_id)
        if workload is None:
            return False
        if not workload.migration_allowed:
            return False
        if workload.region_id == target_region_id:
            return False
        if target_region_id not in cluster.regions:
            return False

        cfg = self._serving_config or None
        migstate = workload.migration

        # PodDisruptionBudget / governor / thermal veto — block BEFORE mutating.
        if migstate is not None:
            veto = self._migration_veto(
                workload, target_region_id, respect_governor=respect_governor
            )
            if veto is not None:
                migstate.migration.veto_count += 1
                migstate.migration.last_veto_reason = veto
                return False

        old_region = workload.region_id
        old_gpu_ids = list(workload.gpu_ids)

        # Release old GPUs
        for region in cluster.regions.values():
            for node in region.nodes:
                for gpu in node.gpus:
                    if gpu.gpu_id in old_gpu_ids:
                        gpu.assigned_workload_id = None
                        gpu.memory_used_bytes = int(gpu.profile.memory_total_bytes * 0.01)

        # Cache-aware cold-reroute cost: the destination cannot reuse the prefix
        # it never cached, so the previously reused prefix tokens must be
        # re-prefilled. Price that lost prefill into a pending TTFT penalty and
        # reset locality confidence — this is what makes naive rerouting destroy
        # TTFT and lets affinity preservation beat cheaper energy.
        cache = workload.cache
        hit_before = cache.prefix.hit_rate if cache is not None else 0.0
        if cache is not None:
            shared = cache.prefix.shared_prefix_tokens or (
                workload.prefix_overlap * workload.avg_seq_len_tokens
            )
            lost = kvc.lost_prefill_tokens(shared, hit_before)
            cache.affinity.cold_route_penalty_ms = kvc.cold_route_penalty_ms(lost, cfg)
            cache.affinity.cold_reroute_count += 1
            cache.affinity.cold_warmup_ticks_remaining = _COLD_START_WARMUP_TICKS
            cache.locality.confidence = kv_value("cold_route_confidence", cfg)
            cache.warmup.warm = False
            cache.warmup.ticks_warm = 0
            cache.prefix.hit_rate = 0.0

        # Update workload region
        workload.region_id = target_region_id
        workload.gpu_ids = []
        workload.node_ids = []
        workload.cold_start_warmup_ticks_remaining = _COLD_START_WARMUP_TICKS
        workload.last_migrated_tick = cluster.tick
        workload.prefix_cache_hit_rate_frac = 0.05   # cold cache after migration

        # Place on new region
        self._place_workload(workload, cluster)

        # The cache is now warm nowhere; its new home is the destination route.
        if cache is not None:
            cache.routing.home_region = target_region_id
            cache.routing.home_gpu_ids = tuple(workload.gpu_ids)
            cache.active_seqs_prev = 0.0

        # Recompute topology score
        workload.topology_score = self._compute_topology_score(workload, cluster)

        # Apply the full migration cost (drain + cold start + cache loss + batch
        # disruption + tail uplift) to the migration state. This is the heart of
        # the realism upgrade: a migration injects a startup TTFT penalty over a
        # warmup window, fragments batching, and amplifies p95/p99.
        self._apply_migration_cost(workload, old_region, target_region_id)

        # Migration is NOT free: drained in-flight requests + rebalancing land as
        # a backlog spike on the destination queue (queue disruption). Combined
        # with the cold cache + warmup set above, this can make aggressive
        # migration LOSE on p99/queue even when it lowers energy cost.
        target_region = cluster.regions.get(target_region_id)
        if target_region is not None:
            disruption = serving_value("migration_queue_disruption")
            for q in target_region.queues:
                if q.service_id == workload.service_id:
                    spike = int(q.arrival_rate_per_sec * 3600 * cluster.tick_duration_hours
                                * 0.01 * disruption)
                    q.queue_depth = min(50000, q.queue_depth + max(0, spike))

        # Log migration
        cluster.migration_log.append({
            "tick": cluster.tick,
            "workload_id": workload_id,
            "from_region": old_region,
            "to_region": target_region_id,
            "old_gpu_ids": old_gpu_ids,
            "new_gpu_ids": list(workload.gpu_ids),
        })
        cluster.migration_count += 1

        return True

    # ------------------------------------------------------------------
    # Migration realism helpers
    # ------------------------------------------------------------------

    def _migration_veto(
        self, workload: SimWorkload, target_region_id: Optional[str] = None,
        *, respect_governor: bool,
    ) -> Optional[str]:
        """Return a veto reason if this migration should be blocked, else None.

        PDB unavailability ALWAYS blocks (drain stall). The remaining governor
        checks (incl. the thermal veto on migrating INTO a hot zone) only apply
        when ``respect_governor`` is set (cache/thermal-aware policies); naive
        policies pass them by and pay the realistic cost instead.
        """
        m = workload.migration
        if m is None:
            return None
        if mig.pdb_blocks_migration(m.pdb.available):
            return "pdb_unavailable"
        if not respect_governor:
            return None
        # Thermal governor: veto migrating INTO a hot destination zone.
        if target_region_id is not None and self._dest_zone_too_hot(target_region_id):
            return "thermal_hot_destination"
        cluster = self._cluster
        cfg = self._serving_config or None
        # Aggregate queue depth for this workload's service in its current region.
        region = cluster.regions.get(workload.region_id)
        qdepth = 0.0
        p95_unstable = False
        if region is not None:
            for q in region.queues:
                if q.service_id == workload.service_id:
                    qdepth = max(qdepth, float(q.queue_depth))
                    sla = workload.queue_sla_p95_ms or 1000.0
                    if q.queue_wait_p95_ms is not None and q.queue_wait_p95_ms > sla:
                        p95_unstable = True
        loc_conf = workload.cache.locality.confidence if workload.cache else 0.0
        prof = mig.resolve_engine_profile(m.engine_runtime)
        return mig.migration_veto_reason(
            queue_depth=qdepth,
            locality_confidence=loc_conf,
            p95_unstable=p95_unstable,
            rollout_instability=m.rollout.instability,
            pdb_available=m.pdb.available,
            warmup_incomplete=m.warmup.ticks_remaining > 0,
            startup_heavy=prof.compile_heavy,
            scale_from_zero=m.coldstart.scale_from_zero,
            config=cfg,
        )

    def _dest_zone_too_hot(self, target_region_id: str) -> bool:
        """True if every rack in the destination region is thermally hot.

        Uses the thermal governor: if the coolest available rack is still above
        the hot-veto temperature (or strongly hotspotted), migrating in is unsafe.
        Missing thermal telemetry lowers the effective threshold (≠ safe).
        """
        cfg = self._serving_config or None
        region = self._cluster.regions.get(target_region_id)
        if region is None:
            return False
        any_rack = False
        for node in region.nodes:
            rt = node.rack_thermal
            if rt is None:
                continue
            any_rack = True
            if not therm.thermal_migration_blocked(
                rt.peak_gpu_temp_c, rt.hotspot.severity, rt.telemetry.tier, cfg
            ):
                return False  # found a cool-enough rack → not blocked
        return any_rack  # all racks hot (and at least one existed)

    def _apply_migration_cost(
        self, workload: SimWorkload, old_region: str, target_region_id: str
    ) -> None:
        """Compute C_mig via migration.py and write it into the migration state."""
        m = workload.migration
        if m is None:
            return
        cfg = self._serving_config or None
        cluster = self._cluster

        # Cross-region RTT, if the scenario configured it.
        rtt_ms = None
        src = cluster.regions.get(old_region)
        if src is not None and target_region_id in src.network_latency_to:
            rtt_ms = float(src.network_latency_to[target_region_id])

        hit_before = workload.cache.prefix.hit_rate if workload.cache else 0.0
        prefill_cost = kv_value("prefill_cost_per_token_ms", cfg)
        from_zero = len(workload.gpu_ids) == 0 or m.coldstart.scale_from_zero
        cohort_eff = m.cohort.efficiency

        cost = mig.migration_cost(
            m.engine_runtime,
            prompt_tokens=workload.avg_seq_len_tokens,
            hit_rate_before=hit_before,
            prefill_cost_per_token_ms=prefill_cost,
            rng=self._rng,
            base_batch_efficiency=cohort_eff,
            churn_rate=m.route_shift.churn_rate,
            rollout_instability=m.rollout.instability,
            queue_pressure=0.0,
            network_rtt_ms=rtt_ms,
            from_zero=from_zero,
            config=cfg,
        )

        # Warm pool absorbs the cold start: a pre-loaded replica skips
        # transfer+warmup, leaving only requeue/routing cost.
        warm = m.warm_pool.size > 0
        startup_penalty = cost.t_requeue_ms + (
            0.0 if warm else cost.t_transfer_ms + cost.t_warmup_ms
        )
        total_startup_s = (cost.drain_s
                           + (0.0 if warm else cost.cold_start.total_seconds))

        # Startup state + warmup window.
        m.startup.last_cold_seconds = cost.cold_start.total_seconds
        m.startup.t_node = cost.cold_start.t_node
        m.startup.t_pull = cost.cold_start.t_pull
        m.startup.t_load = cost.cold_start.t_load
        m.startup.t_gpu_transfer = cost.cold_start.t_gpu_transfer
        m.startup.t_warmup = cost.cold_start.t_warmup
        m.startup.first_compile = cost.cold_start.first_compile
        m.coldstart.cold_start_count += 0 if warm else 1

        warmup_ticks = mig.seconds_to_warmup_ticks(
            total_startup_s, cluster.tick_duration_hours
        )
        m.warmup.ticks_remaining = max(m.warmup.ticks_remaining, warmup_ticks)
        m.warmup.startup_penalty_ms = startup_penalty
        m.warmup.warm = False

        # Drain / eviction bookkeeping.
        m.drain.draining = False
        m.drain.last_drain_seconds = cost.drain_s
        m.drain.drain_seconds_total += cost.drain_s
        m.eviction.last_tick_evictions += 1
        m.eviction.cumulative_evictions += 1

        # Route churn + tail instability spike.
        m.route_shift.reroute_count += 1
        m.route_shift.churn_rate += 1.0
        m.tail.uplift_mult = max(m.tail.uplift_mult, cost.t_tail_mult)
        m.cohort.efficiency = cost.t_batchloss_factor

        # Top-level bookkeeping.
        m.migration.migration_count += 1
        m.migration.last_cost_ms = cost.startup_penalty_ms

    # ------------------------------------------------------------------
    # Phased rollout + governor (public; used by tests / cache-aware policies)
    # ------------------------------------------------------------------

    def can_migrate(self, service_id: str, region_id: Optional[str] = None) -> Optional[str]:
        """Governor check: return a veto reason if migration is unsafe, else None."""
        wl = self._resolve_workload(service_id, region_id)
        if wl is None:
            return "workload_not_found"
        return self._migration_veto(wl, respect_governor=True)

    def safe_migrate_workload(self, workload_id: str, target_region_id: str) -> bool:
        """Governor-respecting migration: vetoes unsafe moves (do-nothing safer)."""
        return self.migrate_workload(workload_id, target_region_id, respect_governor=True)

    def migrate_workload_phased(self, workload_id: str, target_region_id: str) -> bool:
        """Begin/advance a phased (canary) rollout of a cross-region migration.

        Traffic shifts in stabilization-gated phases (0.1→0.25→0.5→1.0). A phase
        advances only when stable; p99 blowups trigger rollback. The first call
        starts the rollout (and performs the underlying placement migration with
        its cost); subsequent calls advance or roll back based on current p99.
        Returns True while the rollout is progressing, False on rollback/block.
        """
        cluster = self._cluster
        wl = cluster.workloads.get(workload_id)
        if wl is None or wl.migration is None:
            return False
        m = wl.migration
        cfg = self._serving_config or None

        if not m.rollout.active:
            # Start the rollout: perform the placement migration once.
            if not self.migrate_workload(workload_id, target_region_id):
                return False
            m.rollout.active = True
            m.rollout.phase = 1
            m.traffic_shift.fraction = 0.1
            m.rollout.hold_ticks_remaining = int(migration_value("rollout_hold_ticks", cfg))
            return True

        # Advancing an in-flight rollout: check stability / rollback.
        region = cluster.regions.get(wl.region_id)
        p99 = 0.0
        if region is not None:
            for q in region.queues:
                if q.service_id == wl.service_id and q.latency_p99_ms is not None:
                    p99 = max(p99, q.latency_p99_ms)
        sla = wl.latency_sla_p99_ms or _SLA_P99_DEFAULT_MS
        if mig.should_rollback(p99, sla, cfg):
            m.rollout.rollback_count += 1
            m.rollout.instability = min(1.0, m.rollout.instability + 0.5)
            m.traffic_shift.fraction = max(0.0, m.traffic_shift.fraction - 0.25)
            return False
        if m.rollout.hold_ticks_remaining > 0:
            return True  # still holding/stabilizing this phase
        stable = p99 <= sla
        new_frac = mig.next_traffic_fraction(m.traffic_shift.fraction, stable)
        m.traffic_shift.fraction = new_frac
        m.rollout.phase += 1 if new_frac > m.traffic_shift.fraction - 1e-9 else 0
        m.rollout.hold_ticks_remaining = int(migration_value("rollout_hold_ticks", cfg))
        if new_frac >= 1.0:
            m.rollout.active = False  # rollout complete
        return True

    def set_warm_pool(self, service_id: str, size: int, region_id: Optional[str] = None) -> bool:
        """Configure a warm pool (pre-loaded ready replicas) for a workload."""
        wl = self._resolve_workload(service_id, region_id)
        if wl is None or wl.migration is None:
            return False
        wl.warm_pool_size = max(0, size)
        wl.migration.warm_pool.size = wl.warm_pool_size
        return True

    def set_pdb(self, service_id: str, min_available: int, region_id: Optional[str] = None) -> bool:
        """Set a PodDisruptionBudget floor; min_available ≥ replicas blocks drains."""
        wl = self._resolve_workload(service_id, region_id)
        if wl is None or wl.migration is None:
            return False
        wl.pdb_min_available = max(0, min_available)
        m = wl.migration
        m.pdb.min_available = wl.pdb_min_available
        m.pdb.available = max(0, len(wl.gpu_ids) - wl.pdb_min_available)
        return True

    # ------------------------------------------------------------------
    # Non-migration action application (Mission 3)
    # ------------------------------------------------------------------
    #
    # These let the benchmark apply the FULL set of safe recommendations against
    # simulated state — not only cross-region migrations — so the constraint-aware
    # policy can actually be measured on thermal/queue/utilization/latency
    # scenarios. Each action mutates SimCluster state; the NEXT tick's physics
    # (_update_thermal / _update_queues / _update_cost_accounting) then reflect it.
    #
    # Realism is intentionally CONSERVATIVE; semantics + confidence are documented
    # per action below. None of these mutate a real cluster — they exist only in
    # the simulator/benchmark harness. In real/customer environments Aurelius
    # remains recommendation_only.

    def _resolve_workload(
        self, service_id: str, region_id: Optional[str] = None
    ) -> Optional[SimWorkload]:
        cluster = self._cluster
        if service_id in cluster.workloads:
            wl = cluster.workloads[service_id]
            if region_id is None or wl.region_id == region_id:
                return wl
        for wl in cluster.workloads.values():
            if wl.service_id == service_id and (region_id is None or wl.region_id == region_id):
                return wl
        return None

    def add_replica(self, service_id: str, region_id: Optional[str] = None) -> bool:
        """SCALE_REPLICAS: attach one idle GPU in-region to the workload.

        Real mechanism: horizontal autoscaling (vLLM/Triton/Ray Serve replica or
        K8s HPA) adding serving capacity. Metrics moved: service_rate rises with
        GPU count (engine.py:722) → queue depth and p95 wait fall, p99 latency
        improves. Side effect modeled: extra power draw (cost accounting). NOT
        modeled: replica spin-up latency to readiness (treated as next-tick).
        Realism: MODERATE_CONFIDENCE — capacity↑→queue↓ is sound; the warmup of a
        fresh replica is approximated as immediate. Calibration: real replica
        ready-time and per-replica throughput from pilot autoscaler metrics.
        """
        cluster = self._cluster
        wl = self._resolve_workload(service_id, region_id)
        if wl is None:
            return False
        # Anti-flapping cooldown: a workload cannot scale again within the
        # stabilization window (real autoscalers use scale-down stabilization).
        cooldown = int(serving_value("scale_cooldown_ticks"))
        if wl.last_scaled_tick is not None and (cluster.tick - wl.last_scaled_tick) < cooldown:
            return False
        region = cluster.regions.get(wl.region_id)
        if region is None:
            return False
        scaling_from_zero = len(wl.gpu_ids) == 0
        for node in region.nodes:
            for gpu in node.gpus:
                if gpu.assigned_workload_id is None:
                    gpu.assigned_workload_id = wl.workload_id
                    gpu.memory_used_bytes = wl.memory_required_bytes
                    wl.gpu_ids.append(gpu.gpu_id)
                    if node.node_id not in wl.node_ids:
                        wl.node_ids.append(node.node_id)
                    wl.gpu_count_required = max(wl.gpu_count_required, len(wl.gpu_ids))
                    wl.topology_score = self._compute_topology_score(wl, cluster)
                    # Autoscaling lag: the new replica is not instantly ready
                    # (provision + container start + model load + readiness). The
                    # workload ramps over replica_warmup_ticks before full tput.
                    wl.cold_start_warmup_ticks_remaining = max(
                        wl.cold_start_warmup_ticks_remaining,
                        int(serving_value("replica_warmup_ticks")),
                    )
                    wl.last_scaled_tick = cluster.tick
                    # Scale-up cold start: engine-specific, heavy-tailed; scale-
                    # FROM-ZERO amplifies TTFT (no warm replica to absorb the
                    # queue while the first replica starts).
                    self._apply_scaleup_cost(wl, from_zero=scaling_from_zero)
                    return True
        return False  # no idle GPU available — scaling not possible this tick

    def _apply_scaleup_cost(self, workload: SimWorkload, *, from_zero: bool) -> None:
        """Apply an autoscaling scale-up startup penalty to the migration state."""
        m = workload.migration
        if m is None:
            return
        cfg = self._serving_config or None
        warm = m.warm_pool.size > 0
        scaleup_s = mig.scaleup_seconds(m.engine_runtime, self._rng, cfg, from_zero=from_zero)
        if warm:
            scaleup_s *= 0.2  # warm pool absorbs most of the startup
        m.coldstart.scale_from_zero = from_zero
        m.coldstart.cold_start_count += 0 if warm else 1
        m.startup.last_cold_seconds = scaleup_s
        warmup_ticks = mig.seconds_to_warmup_ticks(scaleup_s, self._cluster.tick_duration_hours)
        m.warmup.ticks_remaining = max(m.warmup.ticks_remaining, warmup_ticks)
        # Scale-from-zero amplifies TTFT while the first replica starts.
        penalty = scaleup_s * 1000.0 * 0.1
        if from_zero and not warm:
            penalty *= migration_value("scale_from_zero_ttft_mult", cfg)
            m.tail.uplift_mult = max(
                m.tail.uplift_mult, migration_value("scale_from_zero_ttft_mult", cfg)
            )
        m.warmup.startup_penalty_ms = max(m.warmup.startup_penalty_ms, penalty)
        m.warmup.warm = False

    def spread_workload(self, service_id: str, region_id: Optional[str] = None) -> bool:
        """SPREAD: move the workload's hottest GPU onto a cooler idle GPU.

        Real mechanism: pod anti-affinity / topology-spread spreading load off a
        hot rack. Metrics moved: the workload's GPUs run cooler → less thermal
        throttling (engine.py:662, 758-763) → lower TPOT/p99; the vacated node's
        power density falls → rack heat decays (engine.py:639-642). Side effect
        modeled: prefers an idle GPU on a DIFFERENT rack that is strictly cooler.
        NOT modeled: in-flight request reshuffle cost. Realism:
        MODERATE_CONFIDENCE — spreading reduces density and heat; exact thermal
        coupling is a proxy. Calibration: real DCIM rack-thermal coupling.
        """
        cluster = self._cluster
        wl = self._resolve_workload(service_id, region_id)
        if wl is None or not wl.gpu_ids:
            return False
        region = cluster.regions.get(wl.region_id)
        if region is None:
            return False
        assigned = self._workload_gpus(wl, cluster)
        if not assigned:
            return False
        hottest = max(assigned, key=lambda g: g.temperature_c)
        hottest_node = next(
            (n for n in region.nodes if any(g.gpu_id == hottest.gpu_id for g in n.gpus)),
            None,
        )
        hottest_rack = hottest_node.rack_id if hottest_node else None
        idle = [
            (gpu, node)
            for node in region.nodes
            for gpu in node.gpus
            if gpu.assigned_workload_id is None
        ]
        if not idle:
            return False
        # Prefer a cooler GPU on a different rack.
        idle.sort(key=lambda gn: (gn[1].rack_id == hottest_rack, gn[0].temperature_c))
        target_gpu, target_node = idle[0]
        if target_gpu.temperature_c >= hottest.temperature_c - 1.0:
            return False  # no meaningfully cooler destination
        hottest.assigned_workload_id = None
        hottest.memory_used_bytes = int(hottest.profile.memory_total_bytes * 0.01)
        target_gpu.assigned_workload_id = wl.workload_id
        target_gpu.memory_used_bytes = wl.memory_required_bytes
        wl.gpu_ids = [g for g in wl.gpu_ids if g != hottest.gpu_id] + [target_gpu.gpu_id]
        wl.node_ids = sorted({
            n.node_id for n in region.nodes for g in n.gpus if g.gpu_id in wl.gpu_ids
        })
        wl.topology_score = self._compute_topology_score(wl, cluster)
        return True

    def defer_flexible_workload(self, service_id: str, region_id: Optional[str] = None) -> bool:
        """DEFER: shed a flexible/batch workload's load this window (off-peak shift).

        Real mechanism: deferring a batch/flexible job to a cheaper/off-peak
        window. Metrics moved: the workload's target utilization drops → lower
        power draw → lower energy cost this tick. Only applied to NON
        latency-sensitive workloads (deferring a live SLA workload is unsafe).
        NOT modeled: makespan extension / deadline tracking. Realism:
        LOW_CONFIDENCE — captures the energy-now reduction but not the deferred
        work's later cost. Calibration: real batch deadline + catch-up dynamics.
        """
        wl = self._resolve_workload(service_id, region_id)
        if wl is None or wl.latency_sensitive:
            return False
        # Shed load this window: drop target utilization toward idle.
        wl.target_util_pct = min(wl.target_util_pct, 10.0)
        return True

    def consolidate_low_priority(self, region_id: str, service_id: Optional[str] = None) -> bool:
        """CONSOLIDATE: power down nodes left fully idle after packing.

        Real mechanism: bin-packing low-priority workloads and scaling idle nodes
        to a low-power state. Metrics moved: fully-idle nodes drop to ~0 power
        (vs 10% idle) → energy savings; mean utilization of remaining GPUs is
        unaffected here. Guarded by the engine: CONSOLIDATE is suppressed when
        thermal/queue is materially active, so this only fires when it is safe.
        NOT modeled: node resume latency, fragmentation repacking cost. Realism:
        LOW_CONFIDENCE — idle-node power-down is real but the savings magnitude is
        a proxy. Calibration: real node idle vs powered-down power draw.
        """
        cluster = self._cluster
        region = cluster.regions.get(region_id)
        if region is None:
            return False
        changed = False
        for node in region.nodes:
            if node.gpus and all(g.assigned_workload_id is None for g in node.gpus):
                for gpu in node.gpus:
                    if gpu.power_watts > 1.0:
                        gpu.power_watts = 0.0  # scale-to-zero idle node
                        gpu.utilization_pct = 0.0
                        changed = True
        return changed

    @property
    def current_tick(self) -> int:
        return self._cluster.tick

    @property
    def cumulative_metrics(self) -> dict[str, Any]:
        """Return cumulative simulation metrics."""
        cluster = self._cluster
        return {
            "total_energy_cost": cluster.total_energy_cost,
            "total_tokens_served": cluster.total_tokens_served,
            "total_energy_kwh": cluster.total_energy_kwh,
            "sla_violations": cluster.sla_violations,
            "migration_count": cluster.migration_count,
            "cost_per_token": (
                cluster.total_energy_cost / cluster.total_tokens_served
                if cluster.total_tokens_served > 0 else None
            ),
            "tokens_per_joule": (
                cluster.total_tokens_served / (cluster.total_energy_kwh * 3_600_000)
                if cluster.total_energy_kwh > 0 else None
            ),
        }


# ---------------------------------------------------------------------------
# Link rank helper
# ---------------------------------------------------------------------------

def _link_rank(link_type: str) -> int:
    """Return rank for sorting link types (higher = better bandwidth)."""
    return {
        "NVSWITCH": 10,
        "NV4": 9,
        "NV2": 8,
        "NV1": 7,
        "PIX": 6,
        "PXB": 5,
        "PHB": 4,
        "NODE": 3,
        "SYS": 2,
        "RACK": 1,
        "REGION": 0,
    }.get(link_type, 0)
