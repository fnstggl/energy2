"""Connector → ClusterState assembler (Mission 1).

The constraint classifier and recommendation engine consume a single canonical
``ClusterState``. Prior to this module the ONLY producer of ``ClusterState`` was
the simulator: the real connectors (Prometheus/DCGM/vLLM/Triton/Ray/Kubernetes/
topology) emitted *leaf* objects (``GPUState``, ``InferenceServiceState``,
``NodeState``, ``TopologyState``, …) that were never aggregated. This module
closes that gap.

Design rules (mirrors the connector contract):
- Missing sources set ``is_partial=True`` and append to ``missing_sources``; they
  are NEVER fabricated as zeros.
- Stale telemetry lowers the affected region's (and the cluster's) confidence and
  is surfaced via ``sample_age_s``.
- NaN/inf scalars from raw inputs become ``None`` (typed leaf objects are already
  range-validated by their models).
- Region/node/GPU/service references are cross-validated; unknown references are
  preserved with degraded confidence and recorded — never silently invented.
- Sandbox provenance propagates: if any input is sandbox, the cluster is sandbox.
- Read-only: no connector is mutated; nothing is executed.

Public API:
    build_cluster_state(*, timestamp, ...) -> ClusterState
    build_cluster_state_from_connectors(config, connectors, timestamp=None) -> ClusterState
"""

from __future__ import annotations

import logging
import math
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

from .models import (
    ClusterState,
    EnergyState,
    GPUState,
    InferenceServiceState,
    NodeState,
    Provenance,
    RegionState,
    ThermalState,
    TopologyState,
)

logger = logging.getLogger(__name__)

# Region key used when a leaf object carries no region and no default is given.
# It is an explicit sentinel (recorded in missing_sources), never a fake region.
UNASSIGNED_REGION = "_unassigned"

# Staleness threshold: telemetry older than this degrades confidence. HEURISTIC.
_STALE_AGE_S = 300.0


def _finite_or_none(x: Optional[float]) -> Optional[float]:
    """Return x unless it is NaN/inf, in which case None."""
    if x is None:
        return None
    try:
        if math.isnan(x) or math.isinf(x):
            return None
    except TypeError:
        return None
    return x


def _worst_confidence(*levels: str) -> str:
    """Return the lowest confidence label among the inputs."""
    order = {"high": 2, "medium": 1, "low": 0}
    worst = min((order.get(level, 0) for level in levels if level), default=2)
    return {2: "high", 1: "medium", 0: "low"}[worst]


def _degrade(confidence: str, *, partial: bool, stale: bool) -> str:
    """Lower a confidence label when the source is partial and/or stale."""
    if stale:
        confidence = _worst_confidence(confidence, "low")
    elif partial:
        confidence = _worst_confidence(confidence, "medium")
    return confidence


class _RegionAccumulator:
    """Mutable scratch space for one region, finalized into a RegionState."""

    def __init__(self, region_id: str) -> None:
        self.region_id = region_id
        self.nodes: dict[str, NodeState] = {}
        self.services: dict[str, InferenceServiceState] = {}
        self.energy: Optional[EnergyState] = None
        self.thermal: Optional[ThermalState] = None
        self.topology: Optional[TopologyState] = None
        self.spare_capacity_pct: Optional[float] = None
        self.max_age_s: float = 0.0
        self.is_sandbox: bool = False
        self.confidence: str = "high"
        self.source: str = "assembler"

    def note_provenance(self, prov: Optional[Provenance]) -> None:
        if prov is None:
            return
        if prov.is_sandbox:
            self.is_sandbox = True
        if prov.sample_age_s is not None:
            self.max_age_s = max(self.max_age_s, prov.sample_age_s)
        self.confidence = _worst_confidence(self.confidence, prov.confidence)

    def ensure_node(self, node_id: str, timestamp: datetime) -> NodeState:
        node = self.nodes.get(node_id)
        if node is None:
            # Synthesize a minimal node from the GPU's own (region, node_id).
            # This is DERIVED from the GPU, not fabricated capacity data.
            node = NodeState(
                node_id=node_id,
                region=self.region_id,
                timestamp=timestamp,
                provenance=Provenance(
                    source="assembler:derived-node",
                    fetched_at=timestamp,
                    confidence="medium",
                ),
                gpus={},
            )
            self.nodes[node_id] = node
        return node

    def add_gpu(self, gpu: GPUState, timestamp: datetime) -> None:
        node = self.ensure_node(gpu.node_id, timestamp)
        merged = dict(node.gpus)
        merged[gpu.gpu_uuid] = gpu
        self.nodes[node.node_id] = replace(node, gpus=merged)
        self.note_provenance(gpu.provenance)

    def finalize(self, timestamp: datetime) -> RegionState:
        stale = self.max_age_s > _STALE_AGE_S
        confidence = _degrade(self.confidence, partial=False, stale=stale)
        prov = Provenance(
            source=self.source,
            fetched_at=timestamp,
            confidence=confidence,
            is_sandbox=self.is_sandbox,
            sample_age_s=self.max_age_s if self.max_age_s > 0 else None,
        )
        return RegionState(
            region=self.region_id,
            timestamp=timestamp,
            provenance=prov,
            nodes=dict(self.nodes),
            services=dict(self.services),
            energy=self.energy,
            thermal=self.thermal,
            topology=self.topology,
            spare_capacity_pct=self.spare_capacity_pct,
        )


def _coerce_region(region: Optional[str], default_region: Optional[str]) -> tuple[str, bool]:
    """Resolve a leaf object's region. Returns (region_id, is_unassigned)."""
    if region:
        return region, False
    if default_region:
        return default_region, False
    return UNASSIGNED_REGION, True


def _iter_states(obj: Any) -> Iterable[Any]:
    """Yield items from a list/tuple, dict values, or a single object."""
    if obj is None:
        return
    if isinstance(obj, dict):
        yield from obj.values()
    elif isinstance(obj, (list, tuple, set)):
        yield from obj
    else:
        yield obj


def build_cluster_state(
    *,
    timestamp: datetime,
    regions: Optional[dict[str, RegionState]] = None,
    prometheus_snapshot: Any = None,
    gpu_states: Optional[Iterable[GPUState]] = None,
    inference_services: Optional[Iterable[InferenceServiceState]] = None,
    node_states: Any = None,
    workload_states: Any = None,
    queue_states: Any = None,
    topology_state: Any = None,
    energy_states: Any = None,
    thermal_states: Any = None,
    placement_states: Any = None,
    source_metadata: Optional[dict[str, dict[str, Any]]] = None,
    default_region: Optional[str] = None,
    config_hash: Optional[str] = None,
) -> ClusterState:
    """Aggregate connector leaf objects into a single canonical ``ClusterState``.

    All inputs are optional. Absent or failed sources widen ``is_partial`` /
    ``missing_sources`` and degrade confidence rather than fabricating values.

    Parameters
    ----------
    timestamp: snapshot time (UTC-aware required).
    regions: optional pre-built ``{region_id: RegionState}`` to seed and merge into.
    prometheus_snapshot: a ``TelemetrySnapshot`` (used for sandbox/staleness/source
        metadata only; its metrics should already have been normalized into the
        leaf lists by the DCGM/vLLM/etc. adapters).
    gpu_states: ``GPUState`` objects (grouped into region → node).
    inference_services: ``InferenceServiceState`` objects (grouped by region).
    node_states: ``NodeState`` objects, a ``{node_id: NodeState}`` dict, or a
        ``K8sPlacementSnapshot`` (its ``.nodes`` are used).
    topology_state: a ``TopologyState``, or ``{region: TopologyState}``.
    energy_states / thermal_states: ``{region: state}`` dicts or iterables of
        region-tagged states.
    placement_states: a ``K8sPlacementSnapshot`` (or iterable); contributes nodes
        and partiality.
    source_metadata: ``{source_name: {"present": bool, "stale": bool,
        "age_s": float, "error": str}}`` describing connector health.
    default_region: region to assign leaf objects whose region is ``None``.
    config_hash: sha256 of the active config at snapshot time.
    """
    if timestamp.tzinfo is None:
        raise ValueError("build_cluster_state: timestamp must be UTC-aware")

    acc: dict[str, _RegionAccumulator] = {}
    missing_sources: list[str] = []
    unknown_refs: list[str] = []
    cluster_sandbox = False

    def region_acc(region_id: str) -> _RegionAccumulator:
        if region_id not in acc:
            acc[region_id] = _RegionAccumulator(region_id)
        return acc[region_id]

    # --- Seed from pre-built regions -------------------------------------
    for region_id, rstate in (regions or {}).items():
        ra = region_acc(region_id)
        ra.nodes.update(rstate.nodes)
        ra.services.update(rstate.services)
        ra.energy = ra.energy or rstate.energy
        ra.thermal = ra.thermal or rstate.thermal
        ra.topology = ra.topology or rstate.topology
        if rstate.spare_capacity_pct is not None:
            ra.spare_capacity_pct = rstate.spare_capacity_pct
        ra.note_provenance(rstate.provenance)

    # --- Sandbox / staleness from a prometheus snapshot ------------------
    if prometheus_snapshot is not None:
        if getattr(prometheus_snapshot, "is_sandbox", False):
            cluster_sandbox = True

    # --- Node states (incl. K8sPlacementSnapshot) ------------------------
    def absorb_nodes(obj: Any, label: str) -> None:
        nonlocal cluster_sandbox
        if obj is None:
            return
        # K8sPlacementSnapshot: has .nodes dict and partiality flags.
        snap_nodes = getattr(obj, "nodes", None)
        if snap_nodes is not None and not isinstance(obj, (NodeState,)):
            if getattr(obj, "is_partial", False):
                missing_sources.append(f"{label}:partial")
            for src in getattr(obj, "missing_sources", []) or []:
                missing_sources.append(f"{label}:{src}")
            if getattr(obj, "is_sandbox", False):
                cluster_sandbox = True
            node_iter: Iterable[NodeState] = snap_nodes.values()
        else:
            node_iter = _iter_states(obj)
        for node in node_iter:
            if not isinstance(node, NodeState):
                continue
            ra = region_acc(node.region or default_region or UNASSIGNED_REGION)
            if node.region is None and default_region is None:
                unknown_refs.append(f"node[{node.node_id}].region=None")
            existing = ra.nodes.get(node.node_id)
            if existing is not None and existing.gpus and not node.gpus:
                # Preserve GPUs already attached to a synthesized/earlier node.
                ra.nodes[node.node_id] = replace(node, gpus=existing.gpus)
            else:
                ra.nodes[node.node_id] = node
            ra.note_provenance(node.provenance)

    absorb_nodes(node_states, "kubernetes")
    absorb_nodes(placement_states, "kubernetes")

    # --- GPU states ------------------------------------------------------
    for gpu in _iter_states(gpu_states):
        if not isinstance(gpu, GPUState):
            continue
        region_id, unassigned = _coerce_region(gpu.region, default_region)
        if unassigned:
            unknown_refs.append(f"gpu[{gpu.gpu_uuid[:8]}].region=None")
        region_acc(region_id).add_gpu(gpu, timestamp)
        if gpu.provenance.is_sandbox:
            cluster_sandbox = True

    # --- Inference services ----------------------------------------------
    for svc in _iter_states(inference_services):
        if not isinstance(svc, InferenceServiceState):
            continue
        region_id, unassigned = _coerce_region(svc.region, default_region)
        if unassigned:
            unknown_refs.append(f"service[{svc.service_id}].region=None")
        region_acc(region_id).services[svc.service_id] = svc
        region_acc(region_id).note_provenance(svc.provenance)
        if svc.provenance.is_sandbox:
            cluster_sandbox = True

    # --- Energy / thermal (keyed by region) ------------------------------
    def absorb_region_state(obj: Any, attr: str) -> None:
        nonlocal cluster_sandbox
        if obj is None:
            return
        if isinstance(obj, dict):
            items = obj.items()
        else:
            items = ((getattr(s, "region", None), s) for s in _iter_states(obj))
        for region_id, state in items:
            if state is None:
                continue
            rid = region_id or getattr(state, "region", None) or default_region
            if rid is None:
                unknown_refs.append(f"{attr}.region=None")
                rid = UNASSIGNED_REGION
            ra = region_acc(rid)
            setattr(ra, attr, state)
            ra.note_provenance(getattr(state, "provenance", None))
            if getattr(getattr(state, "provenance", None), "is_sandbox", False):
                cluster_sandbox = True

    absorb_region_state(energy_states, "energy")
    absorb_region_state(thermal_states, "thermal")

    # --- Topology --------------------------------------------------------
    if topology_state is not None:
        if isinstance(topology_state, dict):
            for region_id, topo in topology_state.items():
                if topo is not None:
                    region_acc(region_id).topology = topo
        else:
            for topo in _iter_states(topology_state):
                if not isinstance(topo, TopologyState):
                    continue
                # Resolve the topology's node_id to a region; else default/unknown.
                resolved = None
                for ra in acc.values():
                    if topo.node_id in ra.nodes:
                        resolved = ra.region_id
                        break
                if resolved is None:
                    resolved = default_region or UNASSIGNED_REGION
                    if default_region is None:
                        unknown_refs.append(f"topology[{topo.node_id}].region=unresolved")
                region_acc(resolved).topology = topo

    # --- Spare capacity from workload/queue/placement hints --------------
    # (Optional inputs; recorded only if explicitly provided as region dicts.)
    for hint, _name in ((workload_states, "workload"), (queue_states, "queue")):
        if isinstance(hint, dict):
            for region_id, val in hint.items():
                spare = None
                if isinstance(val, (int, float)):
                    spare = _finite_or_none(float(val))
                if spare is not None:
                    region_acc(region_id).spare_capacity_pct = spare

    # --- Source metadata: partiality, staleness --------------------------
    any_stale = False
    if source_metadata:
        for name, meta in source_metadata.items():
            present = meta.get("present", True)
            error = meta.get("error")
            if not present or error:
                missing_sources.append(name if not error else f"{name}:{error}")
            if meta.get("stale"):
                any_stale = True
            age = _finite_or_none(meta.get("age_s"))
            if age is not None and age > _STALE_AGE_S:
                any_stale = True

    # --- Cross-validation: services referencing region with no nodes -----
    for region_id, ra in acc.items():
        if ra.services and not ra.nodes and region_id != UNASSIGNED_REGION:
            unknown_refs.append(f"region[{region_id}]:services_without_nodes")

    if UNASSIGNED_REGION in acc:
        missing_sources.append("unassigned_region_present")

    is_partial = bool(missing_sources) or bool(unknown_refs)
    if unknown_refs:
        missing_sources.extend(unknown_refs)

    # --- Finalize regions ------------------------------------------------
    final_regions: dict[str, RegionState] = {}
    cluster_confidence = "high"
    cluster_max_age = 0.0
    for region_id, ra in acc.items():
        rstate = ra.finalize(timestamp)
        final_regions[region_id] = rstate
        cluster_confidence = _worst_confidence(cluster_confidence, rstate.provenance.confidence)
        if rstate.provenance.sample_age_s:
            cluster_max_age = max(cluster_max_age, rstate.provenance.sample_age_s)

    cluster_confidence = _degrade(cluster_confidence, partial=is_partial, stale=any_stale)

    provenance = Provenance(
        source="assembler",
        fetched_at=timestamp,
        confidence=cluster_confidence,
        is_sandbox=cluster_sandbox,
        sample_age_s=cluster_max_age if cluster_max_age > 0 else None,
    )

    # De-duplicate missing_sources while preserving order.
    seen: set[str] = set()
    deduped = [s for s in missing_sources if not (s in seen or seen.add(s))]

    return ClusterState(
        timestamp=timestamp,
        provenance=provenance,
        regions=final_regions,
        is_partial=is_partial,
        missing_sources=deduped,
        config_hash=config_hash,
    )


def build_cluster_state_from_connectors(
    config: Optional[dict[str, Any]],
    connectors: dict[str, Any],
    timestamp: Optional[datetime] = None,
) -> ClusterState:
    """Drive live/fixture connectors and assemble a ``ClusterState``.

    ``connectors`` is a mapping that may contain any of::

        {
          "prometheus": PrometheusTelemetryConnector,   # .fetch_snapshot()
          "dcgm":       DCGMAdapter,                     # .normalize_gpus(snapshot)
          "vllm":       VLLMAdapter,                     # .normalize_services(snapshot)
          "triton":     TritonAdapter,                   # .normalize_services(snapshot)
          "ray":        RayServeAdapter,                 # .normalize_services(snapshot)
          "kubernetes": KubernetesConnector,             # .fetch_placement()/.collect()
          "topology":   <collector with .collect()>,     # -> TopologyState
          "energy":     {region: EnergyState} | callable,
          "thermal":    {region: ThermalState} | callable,
        }

    Each connector is wrapped in try/except: a failure marks that source missing
    (``is_partial=True``) rather than aborting the whole assembly.
    """
    timestamp = timestamp or datetime.now(tz=timezone.utc)
    config = config or {}
    default_region = config.get("default_region")

    source_metadata: dict[str, dict[str, Any]] = {}
    snapshot = None
    gpu_states: list[GPUState] = []
    services: list[InferenceServiceState] = []
    node_states: list[NodeState] = []
    placement = None
    topology = None
    energy_states = config.get("energy_states")
    thermal_states = config.get("thermal_states")

    def _try(name: str, fn):
        try:
            result = fn()
            source_metadata[name] = {"present": True}
            return result
        except Exception as exc:  # noqa: BLE001 — connector failures must not abort
            logger.warning("connector %s failed during assembly: %s", name, type(exc).__name__)
            source_metadata[name] = {"present": False, "error": type(exc).__name__}
            return None

    prom = connectors.get("prometheus")
    if prom is not None:
        snapshot = _try("prometheus", lambda: prom.fetch_snapshot())

    if snapshot is not None:
        for name in ("dcgm",):
            adapter = connectors.get(name)
            if adapter is not None:
                gpus = _try(name, lambda a=adapter: a.normalize_gpus(snapshot))
                if gpus:
                    gpu_states.extend(gpus)
        for name in ("vllm", "triton", "ray"):
            adapter = connectors.get(name)
            if adapter is not None:
                svcs = _try(name, lambda a=adapter: a.normalize_services(snapshot))
                if svcs:
                    services.extend(svcs)

    k8s = connectors.get("kubernetes")
    if k8s is not None:
        fetch = getattr(k8s, "fetch_placement", None) or getattr(k8s, "collect", None)
        if fetch is not None:
            placement = _try("kubernetes", fetch)

    topo_collector = connectors.get("topology")
    if topo_collector is not None:
        collect = getattr(topo_collector, "collect", None)
        if collect is not None:
            topology = _try("topology", collect)

    return build_cluster_state(
        timestamp=timestamp,
        prometheus_snapshot=snapshot,
        gpu_states=gpu_states or None,
        inference_services=services or None,
        node_states=node_states or None,
        placement_states=placement,
        topology_state=topology,
        energy_states=energy_states,
        thermal_states=thermal_states,
        source_metadata=source_metadata,
        default_region=default_region,
        config_hash=config.get("config_hash"),
    )
