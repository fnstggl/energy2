"""Kubernetes cluster state connector for Aurelius.

Phase 4: read-only K8s API connector that ingests node/pod state
and normalizes it into NodeState and pod placement snapshots.

Design rules:
- default: read-only, no mutations
- real + fake implementations sharing identical normalization paths
- lazy kubernetes client import (optional extra: pip install kubernetes)
- namespace allowlist for multi-tenant safety
- missing/partial data → None or is_partial=True, never fabricated
- secrets never logged

Required read-only RBAC (ClusterRole):
  resources: [nodes, nodes/status, pods]
  verbs: [get, list, watch]

GPU resources in K8s are expressed via nvidia.com/gpu in resource limits only.
GPU capacity/utilization comes from DCGM, not from Kubernetes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from ..state.models import NodeState, Provenance

logger = logging.getLogger(__name__)

# Lazy import guard — kubernetes is an optional extra
_K8S_AVAILABLE = False
try:
    import kubernetes  # noqa: F401
    _K8S_AVAILABLE = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class KubernetesConnectorConfig:
    """Configuration for the Kubernetes read-only connector.

    Attributes:
        kubeconfig_path:      path to kubeconfig; None = default search
        in_cluster:           True = load in-cluster service account token
        namespace_allowlist:  restrict pod collection to these namespaces; empty = all
        timeout_s:            per-request API timeout (seconds)
        is_sandbox:           if True, fixture/fake mode
        region_label:         node label for canonical region
        zone_label:           node label for availability zone
        rack_label:           node label for physical rack
        gpu_product_label:    GFD label for GPU model name
        instance_type_label:  node label for machine/instance type
    """
    kubeconfig_path: Optional[str] = None
    in_cluster: bool = False
    namespace_allowlist: list[str] = field(default_factory=list)
    timeout_s: float = 30.0
    is_sandbox: bool = False
    region_label: str = "topology.kubernetes.io/region"
    zone_label: str = "topology.kubernetes.io/zone"
    rack_label: str = "topology.aurelius.io/rack"
    gpu_product_label: str = "nvidia.com/gpu.product"
    instance_type_label: str = "node.kubernetes.io/instance-type"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "KubernetesConnectorConfig":
        return cls(
            kubeconfig_path=d.get("kubeconfig_path"),
            in_cluster=bool(d.get("in_cluster", False)),
            namespace_allowlist=list(d.get("namespace_allowlist", [])),
            timeout_s=float(d.get("timeout_s", 30.0)),
            is_sandbox=bool(d.get("is_sandbox", False)),
            region_label=d.get("region_label", "topology.kubernetes.io/region"),
            zone_label=d.get("zone_label", "topology.kubernetes.io/zone"),
            rack_label=d.get("rack_label", "topology.aurelius.io/rack"),
            gpu_product_label=d.get("gpu_product_label", "nvidia.com/gpu.product"),
            instance_type_label=d.get("instance_type_label", "node.kubernetes.io/instance-type"),
        )


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------


@dataclass
class PodPlacement:
    """Normalized placement record for a single Kubernetes pod.

    gpu_count: number of GPUs requested in resource limits (0 = no GPU).
    node_name: None means pod is unscheduled/Pending.
    phase:     Pending | Running | Succeeded | Failed | Unknown.
    """
    pod_name: str
    namespace: str
    node_name: Optional[str]
    gpu_count: int
    phase: str
    start_time: Optional[datetime]
    labels: dict[str, str] = field(default_factory=dict)
    owner_kind: Optional[str] = None
    owner_name: Optional[str] = None

    @property
    def is_pending(self) -> bool:
        return self.phase == "Pending" and self.node_name is None

    @property
    def is_running_with_gpu(self) -> bool:
        return self.phase == "Running" and self.gpu_count > 0


@dataclass
class K8sPlacementSnapshot:
    """Result of a single Kubernetes API scrape.

    Attributes:
        nodes:          dict[node_name → NodeState] with GPU capacity + labels
        pods:           normalized pod placements
        fetched_at:     UTC timestamp of the scrape
        is_partial:     True if any API call failed
        missing_sources: names of sub-queries that failed
        is_sandbox:     True if from fixture/fake connector
    """
    nodes: dict[str, NodeState]
    pods: list[PodPlacement]
    fetched_at: datetime
    is_partial: bool = False
    missing_sources: list[str] = field(default_factory=list)
    is_sandbox: bool = False

    @property
    def pending_gpu_pods(self) -> list[PodPlacement]:
        return [p for p in self.pods if p.is_pending and p.gpu_count > 0]

    @property
    def running_gpu_pods(self) -> list[PodPlacement]:
        return [p for p in self.pods if p.is_running_with_gpu]

    def gpu_allocated_per_node(self) -> dict[str, int]:
        """Sum GPU requests for running pods per node."""
        result: dict[str, int] = {}
        for pod in self.pods:
            if pod.phase == "Running" and pod.node_name and pod.gpu_count > 0:
                result[pod.node_name] = result.get(pod.node_name, 0) + pod.gpu_count
        return result


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------


def _parse_gpu_qty(qty_str: Optional[str]) -> Optional[int]:
    """Parse a Kubernetes GPU quantity string ('8', '0', '1') to int.

    Returns None if parsing fails or input is None.
    GPU quantities in K8s are always whole numbers.
    """
    if qty_str is None:
        return None
    try:
        val = int(qty_str)
        return val if val >= 0 else None
    except (ValueError, TypeError):
        return None


def _extract_topology_labels(
    labels: dict[str, str],
    cfg: KubernetesConnectorConfig,
) -> dict[str, Optional[str]]:
    """Extract topology-relevant labels from a K8s node's label dict."""
    return {
        "region": labels.get(cfg.region_label),
        "zone": labels.get(cfg.zone_label),
        "rack_id": labels.get(cfg.rack_label),
        "instance_type": labels.get(cfg.instance_type_label),
        "gpu_product": labels.get(cfg.gpu_product_label),
    }


def normalize_node_dict(
    node_dict: dict[str, Any],
    cfg: KubernetesConnectorConfig,
    ts: datetime,
    allocated_per_node: dict[str, int],
) -> Optional[NodeState]:
    """Normalize a raw K8s V1Node dict into a NodeState.

    Returns None on parse failure — caller populates missing_sources.
    """
    try:
        metadata = node_dict.get("metadata") or {}
        node_name = metadata.get("name") or ""
        labels: dict[str, str] = dict(metadata.get("labels") or {})
        taints_raw = (node_dict.get("spec") or {}).get("taints") or []

        status = node_dict.get("status") or {}
        capacity = status.get("capacity") or {}
        allocatable = status.get("allocatable") or {}
        conditions = status.get("conditions") or []

        topo = _extract_topology_labels(labels, cfg)
        region = topo["region"] or "unknown"

        gpu_capacity = _parse_gpu_qty(capacity.get("nvidia.com/gpu"))
        gpu_allocatable = _parse_gpu_qty(allocatable.get("nvidia.com/gpu"))
        gpu_allocated = allocated_per_node.get(node_name)

        # schedulable = not (spec.unschedulable == True)
        spec = node_dict.get("spec") or {}
        schedulable = not bool(spec.get("unschedulable", False))

        # Override schedulable if Node condition Ready=False
        for cond in conditions:
            if cond.get("type") == "Ready" and cond.get("status") == "False":
                schedulable = False
                break

        # Normalize taints to list[dict]
        taints: list[dict[str, Any]] = []
        for t in taints_raw:
            taints.append({
                "key": t.get("key", ""),
                "value": t.get("value"),
                "effect": t.get("effect", ""),
            })

        prov = Provenance(
            source="kubernetes",
            fetched_at=ts,
            confidence="high",
            is_sandbox=cfg.is_sandbox,
        )

        return NodeState(
            node_id=node_name,
            region=region,
            zone=topo.get("zone"),
            rack_id=topo.get("rack_id"),
            instance_type=topo.get("instance_type"),
            timestamp=ts,
            provenance=prov,
            gpu_capacity=gpu_capacity,
            gpu_allocatable=gpu_allocatable,
            gpu_allocated=gpu_allocated,
            labels=labels,
            taints=taints,
            schedulable=schedulable,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to normalize K8s node dict: %s", exc)
        return None


def normalize_pod_dict(pod_dict: dict[str, Any]) -> Optional[PodPlacement]:
    """Normalize a raw K8s V1Pod dict into a PodPlacement.

    GPU resources are in spec.containers[*].resources.limits["nvidia.com/gpu"].
    The requests field is intentionally ignored — K8s uses limits for GPU isolation.
    Returns None on parse failure.
    """
    try:
        metadata = pod_dict.get("metadata") or {}
        pod_name = metadata.get("name") or ""
        namespace = metadata.get("namespace") or "default"
        labels: dict[str, str] = dict(metadata.get("labels") or {})

        spec = pod_dict.get("spec") or {}
        node_name: Optional[str] = spec.get("nodeName")
        containers = spec.get("containers") or []

        # Sum GPU limits across all containers
        gpu_count = 0
        for container in containers:
            resources = container.get("resources") or {}
            limits = resources.get("limits") or {}
            qty = _parse_gpu_qty(limits.get("nvidia.com/gpu"))
            if qty:
                gpu_count += qty

        status = pod_dict.get("status") or {}
        phase = status.get("phase") or "Unknown"

        start_time: Optional[datetime] = None
        start_time_str = status.get("startTime")
        if start_time_str:
            try:
                start_time = datetime.fromisoformat(
                    start_time_str.replace("Z", "+00:00")
                )
            except (ValueError, AttributeError):
                pass

        # First owner reference
        owner_kind = None
        owner_name = None
        owners = metadata.get("ownerReferences") or []
        if owners:
            owner_kind = owners[0].get("kind")
            owner_name = owners[0].get("name")

        return PodPlacement(
            pod_name=pod_name,
            namespace=namespace,
            node_name=node_name,
            gpu_count=gpu_count,
            phase=phase,
            start_time=start_time,
            labels=labels,
            owner_kind=owner_kind,
            owner_name=owner_name,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to normalize K8s pod dict: %s", exc)
        return None


def _build_snapshot(
    raw_nodes: list[dict[str, Any]],
    raw_pods: list[dict[str, Any]],
    cfg: KubernetesConnectorConfig,
    ts: datetime,
    initial_missing: list[str],
) -> K8sPlacementSnapshot:
    """Build a K8sPlacementSnapshot from raw K8s API dicts.

    Shared by KubernetesConnector and FakeKubernetesConnector so both
    use identical normalization code paths.
    """
    missing = list(initial_missing)

    # Normalize pods first (needed to compute gpu_allocated_per_node)
    pods: list[PodPlacement] = []
    for pod_dict in raw_pods:
        p = normalize_pod_dict(pod_dict)
        if p is not None:
            pods.append(p)

    # Compute GPU allocation per node from running pods
    allocated_per_node: dict[str, int] = {}
    for pod in pods:
        if pod.phase == "Running" and pod.node_name and pod.gpu_count > 0:
            allocated_per_node[pod.node_name] = (
                allocated_per_node.get(pod.node_name, 0) + pod.gpu_count
            )

    # Normalize nodes
    nodes: dict[str, NodeState] = {}
    for node_dict in raw_nodes:
        n = normalize_node_dict(node_dict, cfg, ts, allocated_per_node)
        if n is not None:
            nodes[n.node_id] = n
        else:
            node_name = (node_dict.get("metadata") or {}).get("name", "?")
            missing.append(f"kubernetes-node-parse:{node_name}")

    return K8sPlacementSnapshot(
        nodes=nodes,
        pods=pods,
        fetched_at=ts,
        is_partial=bool(missing),
        missing_sources=missing,
        is_sandbox=cfg.is_sandbox,
    )


# ---------------------------------------------------------------------------
# Real connector
# ---------------------------------------------------------------------------


class KubernetesConnector:
    """Read-only Kubernetes API connector.

    Lazily loads the kubernetes client (pip install kubernetes).
    All failures produce partial snapshots with missing_sources populated.
    No cluster mutations are performed.

    Usage:
        cfg = KubernetesConnectorConfig(in_cluster=True)
        connector = KubernetesConnector(cfg)
        snapshot = connector.collect()
    """

    def __init__(self, cfg: Optional[KubernetesConnectorConfig] = None) -> None:
        self._cfg = cfg or KubernetesConnectorConfig()
        self._core_api = None

    def _get_core_api(self) -> Any:
        if self._core_api is None:
            if not _K8S_AVAILABLE:
                raise RuntimeError(
                    "kubernetes package not available. "
                    "Install with: pip install kubernetes"
                )
            from kubernetes import client  # type: ignore
            from kubernetes import config as k8s_config
            if self._cfg.in_cluster:
                k8s_config.load_incluster_config()
            elif self._cfg.kubeconfig_path:
                k8s_config.load_kube_config(config_file=self._cfg.kubeconfig_path)
            else:
                k8s_config.load_kube_config()
            self._core_api = client.CoreV1Api()
        return self._core_api

    def collect(self) -> K8sPlacementSnapshot:
        """Scrape the K8s API and return a placement snapshot.

        On total API failure, returns an empty snapshot with is_partial=True.
        Never raises into the caller.
        """
        ts = datetime.now(tz=timezone.utc)
        missing: list[str] = []
        raw_nodes: list[dict[str, Any]] = []
        raw_pods: list[dict[str, Any]] = []

        try:
            api = self._get_core_api()
        except Exception as exc:
            logger.warning("K8s client init failed: %s", exc)
            return K8sPlacementSnapshot(
                nodes={},
                pods=[],
                fetched_at=ts,
                is_partial=True,
                missing_sources=["kubernetes-client-init"],
                is_sandbox=self._cfg.is_sandbox,
            )

        # List nodes (read-only: get,list,watch on nodes)
        try:
            node_list = api.list_node(
                timeout_seconds=int(self._cfg.timeout_s)
            )
            raw_nodes = [self._node_to_dict(n) for n in node_list.items]
        except Exception as exc:
            logger.warning("K8s list_node failed: %s", exc)
            missing.append("kubernetes-nodes")

        # List pods (read-only: get,list,watch on pods)
        try:
            if self._cfg.namespace_allowlist:
                for ns in self._cfg.namespace_allowlist:
                    pod_list = api.list_namespaced_pod(
                        namespace=ns,
                        timeout_seconds=int(self._cfg.timeout_s),
                    )
                    raw_pods.extend(self._pod_to_dict(p) for p in pod_list.items)
            else:
                pod_list = api.list_pod_for_all_namespaces(
                    timeout_seconds=int(self._cfg.timeout_s)
                )
                raw_pods = [self._pod_to_dict(p) for p in pod_list.items]
        except Exception as exc:
            logger.warning("K8s list_pod failed: %s", exc)
            missing.append("kubernetes-pods")

        return _build_snapshot(raw_nodes, raw_pods, self._cfg, ts, missing)

    @staticmethod
    def _node_to_dict(node: Any) -> dict[str, Any]:
        """Convert a kubernetes V1Node object to a plain dict."""
        metadata = node.metadata or type("", (), {})()
        spec = node.spec or type("", (), {})()
        status = node.status or type("", (), {})()

        taints: list[dict] = []
        for t in (getattr(spec, "taints", None) or []):
            taints.append({
                "key": t.key or "",
                "value": t.value,
                "effect": t.effect or "",
            })

        conditions: list[dict] = []
        for c in (getattr(status, "conditions", None) or []):
            conditions.append({"type": c.type, "status": c.status})

        return {
            "metadata": {
                "name": getattr(metadata, "name", "") or "",
                "labels": dict(getattr(metadata, "labels", {}) or {}),
                "ownerReferences": [],
            },
            "spec": {
                "unschedulable": bool(getattr(spec, "unschedulable", False)),
                "taints": taints,
            },
            "status": {
                "capacity": dict(getattr(status, "capacity", {}) or {}),
                "allocatable": dict(getattr(status, "allocatable", {}) or {}),
                "conditions": conditions,
            },
        }

    @staticmethod
    def _pod_to_dict(pod: Any) -> dict[str, Any]:
        """Convert a kubernetes V1Pod object to a plain dict."""
        metadata = pod.metadata or type("", (), {})()
        spec = pod.spec or type("", (), {})()
        status = pod.status or type("", (), {})()

        containers: list[dict] = []
        for c in (getattr(spec, "containers", None) or []):
            limits: dict = {}
            if c.resources and c.resources.limits:
                limits = dict(c.resources.limits)
            containers.append({"name": c.name, "resources": {"limits": limits}})

        owners: list[dict] = []
        for ref in (getattr(metadata, "owner_references", None) or []):
            owners.append({"kind": ref.kind, "name": ref.name})

        start_time_raw = getattr(status, "start_time", None)
        start_time_str = None
        if start_time_raw is not None:
            try:
                start_time_str = start_time_raw.isoformat()
            except Exception:  # noqa: BLE001
                pass

        return {
            "metadata": {
                "name": getattr(metadata, "name", "") or "",
                "namespace": getattr(metadata, "namespace", "default") or "default",
                "labels": dict(getattr(metadata, "labels", {}) or {}),
                "ownerReferences": owners,
            },
            "spec": {
                "nodeName": getattr(spec, "node_name", None),
                "containers": containers,
            },
            "status": {
                "phase": getattr(status, "phase", "Unknown") or "Unknown",
                "startTime": start_time_str,
            },
        }


# ---------------------------------------------------------------------------
# Fake (sandbox) connector
# ---------------------------------------------------------------------------


class FakeKubernetesConnector:
    """Fixture-based sandbox Kubernetes connector.

    Accepts raw node_list and pod_list dicts mirroring the K8s API dict shape.
    Uses the exact same normalization paths as KubernetesConnector — no
    special-casing or duplicate logic.

    Usage:
        connector = FakeKubernetesConnector(
            node_list=NODE_FIXTURE,
            pod_list=POD_FIXTURE,
        )
        snapshot = connector.collect()
    """

    def __init__(
        self,
        node_list: Optional[list[dict[str, Any]]] = None,
        pod_list: Optional[list[dict[str, Any]]] = None,
        cfg: Optional[KubernetesConnectorConfig] = None,
        is_partial: bool = False,
        missing_sources: Optional[list[str]] = None,
    ) -> None:
        self._raw_nodes = list(node_list or [])
        self._raw_pods = list(pod_list or [])
        self._cfg = cfg or KubernetesConnectorConfig(is_sandbox=True)
        self._is_partial = is_partial
        self._missing_sources = list(missing_sources or [])

    def collect(self) -> K8sPlacementSnapshot:
        ts = datetime.now(tz=timezone.utc)
        initial_missing = list(self._missing_sources) if self._is_partial else []
        return _build_snapshot(
            self._raw_nodes,
            self._raw_pods,
            self._cfg,
            ts,
            initial_missing,
        )
