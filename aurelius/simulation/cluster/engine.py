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

from .model import (
    GPU_PROFILES,
    GPUProfile,
    SimCluster,
    SimGPU,
    SimNode,
    SimQueue,
    SimRegion,
    SimTopologyLink,
    SimWorkload,
    SimulatorConfig,
)
from ...state.models import (
    ClusterState,
    EnergyState,
    GPUState,
    InferenceServiceState,
    NodeState,
    Provenance,
    RegionState,
)

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

        gpus = []
        for idx in range(gpu_count):
            gpu_id = f"{node_id}-gpu{idx}"
            gpu_uuid = f"GPU-{uuid.uuid4().hex[:8].upper()}"
            gpu = SimGPU(
                gpu_id=gpu_id,
                gpu_index=idx,
                uuid=gpu_uuid,
                node_id=node_id,
                profile=profile,
                temperature_c=self._rng.uniform(32.0, 38.0),
                power_watts=profile.base_power_watts * 0.1,  # idle
            )
            gpus.append(gpu)

        # Build topology links
        links = self._build_topology_links(gpus, topology_class)

        node = SimNode(
            node_id=node_id,
            region_id=region_id,
            zone=zone,
            rack_id=rack_id,
            instance_type=node_cfg.get("instance_type", f"gpu.{gpu_count}x{gpu_type}"),
            gpus=gpus,
            topology_links=links,
            labels={
                "topology.kubernetes.io/region": region_id,
                "topology.kubernetes.io/zone": zone,
                "gpu-type": gpu_type,
                "topology-class": topology_class,
            },
        )
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
        )

        # Place workload onto GPUs in the target region
        self._place_workload(workload, cluster)
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
        self._update_queues(cluster)
        self._update_cache_proxy(cluster)
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
            self._update_queues(cluster)
            self._update_cache_proxy(cluster)
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
                        node.rack_heat_delta_c += extra_heat

            elif etype == "thermal_hotspot_end":
                node_id = event.get("node_id")
                for region in cluster.regions.values():
                    for node in region.nodes:
                        if node_id and node.node_id != node_id:
                            continue
                        node.rack_heat_delta_c = 0.0

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
                    wl.kv_cache_usage_frac = cache_usage
                    wl.prefix_cache_hit_rate_frac = hit_rate

            elif etype == "kv_cache_pressure_end":
                service_id = event.get("service_id")
                for wl in cluster.workloads.values():
                    if service_id and wl.service_id != service_id:
                        continue
                    wl.kv_cache_usage_frac = 0.3
                    wl.prefix_cache_hit_rate_frac = 0.5

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

                    # Power: base + (util/100) * (tdp - base)
                    p = gpu.profile.base_power_watts
                    tdp = gpu.profile.max_power_watts
                    gpu.power_watts = p * 0.1 + (util / 100.0) * (tdp - p * 0.1)

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
        """Update GPU temperatures and detect throttling using low-pass filter."""
        for region in cluster.regions.values():
            # Accumulate rack heat from high-power nodes
            for node in region.nodes:
                total_node_power = sum(g.power_watts for g in node.gpus)
                max_node_power = sum(g.profile.max_power_watts for g in node.gpus)
                if max_node_power > 0:
                    power_fraction = total_node_power / max_node_power
                else:
                    power_fraction = 0.0

                # Rack heat builds when load is high, decays when load is low
                if power_fraction > 0.7:
                    node.rack_heat_delta_c += _RACK_HEAT_ALPHA * (power_fraction - 0.7) * 20
                else:
                    node.rack_heat_delta_c = max(0.0, node.rack_heat_delta_c - _RACK_HEAT_DECAY * 5)

                node.rack_heat_delta_c = min(node.rack_heat_delta_c, 25.0)

                ambient = region.ambient_temp_c + node.rack_heat_delta_c

                for gpu in node.gpus:
                    util_frac = gpu.utilization_pct / 100.0
                    # Target temp = ambient + util * (TDP temp rise)
                    # degrees at full load above 20°C ambient
                    tdp_rise = gpu.profile.throttle_temp_c - 20.0
                    t_target = ambient + util_frac * tdp_rise
                    t_target = min(t_target, _MAX_REALISTIC_TEMP_C)

                    # Low-pass filter (EMA)
                    gpu.temperature_c = (
                        _THERMAL_ALPHA * t_target + (1 - _THERMAL_ALPHA) * gpu.temperature_c
                    )

                    # Throttling detection
                    gpu.thermal_throttle_active = gpu.temperature_c > _THROTTLE_TEMP_C

    def _update_queues(self, cluster: SimCluster) -> None:
        """Update queue state using M/M/1 latency approximation."""
        tick = cluster.tick
        # Hour of day for diurnal modulation (0-23)
        hour = (tick - 1) % 24

        for region in cluster.regions.values():
            for queue in region.queues:
                # Arrival rate with diurnal modulation: peak around 14:00
                diurnal_factor = 1.0 + queue.diurnal_amplitude * math.sin(
                    math.pi * (hour - 6) / 12
                )
                diurnal_factor = max(0.1, diurnal_factor)

                arrival_rate = queue.base_arrival_rate_per_sec * diurnal_factor
                if queue.surge_active:
                    arrival_rate *= queue.surge_multiplier

                queue.arrival_rate_per_sec = arrival_rate

                # Service rate: depends on effective GPU throughput
                # Find GPUs serving this queue's service
                workload = self._find_workload_for_service(
                    queue.service_id, region.region_id, cluster
                )
                if workload is None:
                    queue.service_rate_per_sec = 0.01   # effectively no service
                    queue.queue_depth = min(
                        queue.queue_depth + int(arrival_rate * 3600 * 0.1), 10000
                    )
                    queue.queue_wait_p95_ms = 60000.0   # 60s wait
                    continue

                # Effective service rate based on GPU throughput
                gpu_util = self._workload_effective_util(workload, cluster)
                warmup_factor = max(
                    0.2,
                    1.0 - workload.cold_start_warmup_ticks_remaining / _COLD_START_WARMUP_TICKS,
                )

                # Tokens per second per GPU at current utilization
                profile = self._get_workload_gpu_profile(workload, cluster)
                if profile is not None:
                    tokens_per_sec = (
                        profile.tokens_per_sec_at_full_util * (gpu_util / 100.0) * warmup_factor
                    )
                else:
                    tokens_per_sec = 1000.0 * (gpu_util / 100.0)

                # Apply topology penalty for multi-GPU communication-heavy workloads
                topo_penalty = 1.0 - (1.0 - workload.topology_score) * {
                    "low": 0.1,
                    "medium": 0.3,
                    "high": 0.5,
                }.get(workload.communication_intensity, 0.1)
                tokens_per_sec *= topo_penalty

                # Total service rate (requests/sec, assuming _TOKENS_PER_REQUEST tokens each)
                total_tokens_per_sec = tokens_per_sec * len(workload.gpu_ids)
                service_rate = (
                    total_tokens_per_sec / _TOKENS_PER_REQUEST if _TOKENS_PER_REQUEST > 0 else 0.0
                )
                queue.service_rate_per_sec = max(0.01, service_rate)

                workload.effective_tokens_per_second = total_tokens_per_sec
                workload.effective_requests_per_second = service_rate

                # M/M/1: rho = lambda/mu, E[W] = rho / (mu * (1 - rho))
                rho = min(0.99, arrival_rate / queue.service_rate_per_sec)

                # Queue depth update (discrete-time M/M/1 approximation)
                net_arrival = arrival_rate - queue.service_rate_per_sec
                delta = int(net_arrival * 3600 * cluster.tick_duration_hours * 0.01)
                queue.queue_depth = max(0, queue.queue_depth + delta)
                queue.queue_depth = min(queue.queue_depth, 50000)

                # Latency from M/M/1
                if rho < 0.99 and queue.service_rate_per_sec > 0:
                    mean_wait_s = rho / (queue.service_rate_per_sec * (1.0 - rho))
                else:
                    mean_wait_s = 60.0   # saturated

                queue.queue_wait_p95_ms = mean_wait_s * 1000.0 * 3.0  # p95 ≈ 3x mean for M/M/1

                # TTFT: base + queue wait + memory pressure penalty
                cache_penalty = 1.0 + 2.0 * max(0, workload.kv_cache_usage_frac - 0.7)
                ttft_base = _BASE_TTFT_MS / max(0.1, warmup_factor)

                queue.ttft_p50_ms = ttft_base * (1.0 + rho * 2.0) * cache_penalty
                queue.ttft_p95_ms = queue.ttft_p50_ms * 2.5
                queue.ttft_p99_ms = queue.ttft_p50_ms * 5.0

                # TPOT: relatively stable, affected by GPU throttling
                throttle_factor = 1.0
                for gpu in self._workload_gpus(workload, cluster):
                    if gpu.thermal_throttle_active:
                        throttle_factor = max(
                            throttle_factor,
                            1.0 + (gpu.temperature_c - _THROTTLE_TEMP_C) / 20.0,
                        )

                queue.tpot_p50_ms = _BASE_TPOT_MS * throttle_factor
                queue.tpot_p95_ms = queue.tpot_p50_ms * 2.0
                queue.tpot_p99_ms = queue.tpot_p50_ms * 4.0

                # End-to-end latency ≈ TTFT + TPOT * avg_output_tokens
                avg_output_tokens = 128
                queue.latency_p50_ms = queue.ttft_p50_ms + queue.tpot_p50_ms * avg_output_tokens
                queue.latency_p95_ms = queue.ttft_p95_ms + queue.tpot_p95_ms * avg_output_tokens
                queue.latency_p99_ms = queue.ttft_p99_ms + queue.tpot_p99_ms * avg_output_tokens

                # Timeout rate: when p99 > SLA
                sla_ms = workload.latency_sla_p99_ms or _SLA_P99_DEFAULT_MS
                if queue.latency_p99_ms > sla_ms:
                    timeout_rate = min(50.0, (queue.latency_p99_ms - sla_ms) / sla_ms * 10.0)
                    cluster.sla_violations += 1
                else:
                    timeout_rate = 0.0
                queue.timeout_rate_pct = timeout_rate

                # Active sequences and batch size
                queue.active_sequences = min(int(arrival_rate * 0.5), 512)
                queue.batch_size = min(queue.active_sequences, 64)
                queue.tokens_per_second = total_tokens_per_sec
                queue.requests_per_second = service_rate

    def _update_cache_proxy(self, cluster: SimCluster) -> None:
        """Update KV cache and prefix cache hit rate proxies."""
        for region in cluster.regions.values():
            for queue in region.queues:
                workload = self._find_workload_for_service(
                    queue.service_id, region.region_id, cluster
                )
                if workload is None:
                    continue

                # KV cache builds up as memory pressure increases
                mem_used_frac = 0.0
                gpus = self._workload_gpus(workload, cluster)
                if gpus:
                    mem_used_frac = sum(
                        g.memory_used_bytes / g.profile.memory_total_bytes for g in gpus
                    ) / len(gpus)

                workload.kv_cache_usage_frac = max(
                    workload.kv_cache_usage_frac, mem_used_frac * 0.8
                )
                workload.kv_cache_usage_frac = min(workload.kv_cache_usage_frac, 0.98)

                # Prefix cache hit rate decreases after migration (cold start)
                if workload.cold_start_warmup_ticks_remaining > 0:
                    warmup_pct = (
                        workload.cold_start_warmup_ticks_remaining / _COLD_START_WARMUP_TICKS
                    )
                    workload.prefix_cache_hit_rate_frac = max(
                        0.05, workload.prefix_cache_hit_rate_frac * (1.0 - warmup_pct)
                    )

                queue.kv_cache_usage_pct = workload.kv_cache_usage_frac * 100.0
                queue.prefix_cache_hit_rate_pct = workload.prefix_cache_hit_rate_frac * 100.0

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
        )

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

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

        # Communication intensity multiplier: poor topology hurts high-comm more
        comm_weight = {"low": 0.3, "medium": 0.6, "high": 1.0}.get(
            workload.communication_intensity, 0.3
        )
        # Weighted score: high-comm workloads get a stronger penalty for bad topology
        return base_score * (1.0 - comm_weight) + base_score * comm_weight

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

            rs = RegionState(
                region=region.region_id,
                timestamp=ts,
                provenance=prov,
                nodes=node_states,
                services=service_states,
                energy=energy,
                spare_capacity_pct=spare_pct,
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
    ) -> bool:
        """Simulate workload migration to another region.

        This is the benchmark feedback mechanism for optimizer policies.
        Safety checks prevent invalid migrations:
        - workload not found → False
        - migration_allowed=False → False
        - same region → False (no-op)
        - unknown region → False
        - insufficient capacity in target → False

        Returns True if migration was applied, False if blocked.
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

        old_region = workload.region_id
        old_gpu_ids = list(workload.gpu_ids)

        # Release old GPUs
        for region in cluster.regions.values():
            for node in region.nodes:
                for gpu in node.gpus:
                    if gpu.gpu_id in old_gpu_ids:
                        gpu.assigned_workload_id = None
                        gpu.memory_used_bytes = int(gpu.profile.memory_total_bytes * 0.01)

        # Update workload region
        workload.region_id = target_region_id
        workload.gpu_ids = []
        workload.node_ids = []
        workload.cold_start_warmup_ticks_remaining = _COLD_START_WARMUP_TICKS
        workload.last_migrated_tick = cluster.tick
        workload.prefix_cache_hit_rate_frac = 0.05   # cold cache after migration

        # Place on new region
        self._place_workload(workload, cluster)

        # Recompute topology score
        workload.topology_score = self._compute_topology_score(workload, cluster)

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
