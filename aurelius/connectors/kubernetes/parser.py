"""Kubernetes object parser — nodes, pods, workloads → normalized state models.

Parses raw Kubernetes API JSON responses (from live API or fixtures) into
NodeState and WorkloadState instances without requiring a live cluster.

Read-only: this module never creates, updates, or deletes Kubernetes resources.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from ...state.models import (
    CommunicationIntensity,
    NodeState,
    Provenance,
    QueueState,
    WorkloadState,
    WorkloadType,
)

# Topology label keys used by Kubernetes node topology awareness
_REGION_LABEL = "topology.kubernetes.io/region"
_ZONE_LABEL = "topology.kubernetes.io/zone"
_RACK_LABELS = [
    "topology.kubernetes.io/rack",
    "rack",
    "kubernetes.io/rack",
    "failure-domain.beta.kubernetes.io/rack",
]

# GPU resource request keys
_GPU_RESOURCE_KEYS = [
    "nvidia.com/gpu",
    "amd.com/gpu",
    "gpu",
]


@dataclass
class KubernetesParseResult:
    """Normalized output from Kubernetes API parsing."""

    nodes: dict[str, NodeState] = field(default_factory=dict)
    workloads: dict[str, WorkloadState] = field(default_factory=dict)
    pending_queues: dict[str, QueueState] = field(default_factory=dict)
    parse_errors: list[str] = field(default_factory=list)
    collected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class KubernetesParser:
    """Parses Kubernetes API objects into Aurelius state models.

    Works from raw API JSON dicts — no kubernetes Python client required.
    This means the same parser handles both live API responses and fixture files.

    Minimal required RBAC::

        apiVersion: rbac.authorization.k8s.io/v1
        kind: ClusterRole
        metadata:
          name: aurelius-reader
        rules:
        - apiGroups: [""]
          resources: ["nodes", "pods", "namespaces"]
          verbs: ["get", "list", "watch"]

    Parameters
    ----------
    namespace_allowlist:
        If non-empty, only pods in these namespaces are parsed.
        Empty list = all namespaces.
    """

    def __init__(
        self,
        namespace_allowlist: Optional[list[str]] = None,
    ) -> None:
        self.namespace_allowlist = set(namespace_allowlist or [])

    # ------------------------------------------------------------------
    # Node parsing
    # ------------------------------------------------------------------

    def parse_node_list(
        self,
        node_list: dict[str, Any],
        source: str = "kubernetes:nodes",
    ) -> dict[str, NodeState]:
        """Parse a Kubernetes NodeList API response."""
        nodes: dict[str, NodeState] = {}
        collected_at = datetime.now(timezone.utc)

        for item in node_list.get("items", []):
            try:
                node = self._parse_node(item, collected_at, source)
                nodes[node.node_id] = node
            except Exception:
                pass  # individual parse failures don't abort the batch

        return nodes

    def _parse_node(
        self,
        item: dict[str, Any],
        collected_at: datetime,
        source: str,
    ) -> NodeState:
        metadata = item.get("metadata", {})
        name = metadata.get("name", "unknown")
        labels = metadata.get("labels", {})
        spec = item.get("spec", {})
        status = item.get("status", {})

        # Topology
        region_id = labels.get(_REGION_LABEL)
        zone = labels.get(_ZONE_LABEL)
        rack_id = None
        for rack_key in _RACK_LABELS:
            if rack_key in labels:
                rack_id = labels[rack_key]
                break

        # Instance type
        instance_type = (
            labels.get("beta.kubernetes.io/instance-type")
            or labels.get("node.kubernetes.io/instance-type")
        )

        # Taints
        taints = [
            {"key": t.get("key", ""), "effect": t.get("effect", ""), "value": t.get("value", "")}
            for t in spec.get("taints", [])
        ]

        # Readiness
        ready = None
        unschedulable = spec.get("unschedulable", False)
        for condition in status.get("conditions", []):
            if condition.get("type") == "Ready":
                ready = condition.get("status") == "True"
                break

        # Capacity / allocatable
        allocatable = status.get("allocatable", {})
        capacity = status.get("capacity", {})

        gpu_count = self._parse_gpu_count(capacity)
        allocatable_gpu = self._parse_gpu_count(allocatable)
        allocatable_cpu = self._parse_cpu_millicores(allocatable.get("cpu"))
        allocatable_memory = self._parse_bytes(allocatable.get("memory"))

        prov = Provenance(source=source, collected_at=collected_at, confidence=1.0)

        return NodeState(
            node_id=name,
            region_id=region_id,
            zone=zone,
            rack_id=rack_id,
            instance_type=instance_type,
            gpu_count=gpu_count,
            labels=labels,
            taints=taints,
            ready=ready,
            unschedulable=unschedulable or False,
            allocatable_gpu=allocatable_gpu,
            allocatable_cpu_millicores=allocatable_cpu,
            allocatable_memory_bytes=allocatable_memory,
            provenance=prov,
        )

    # ------------------------------------------------------------------
    # Pod / workload parsing
    # ------------------------------------------------------------------

    def parse_pod_list(
        self,
        pod_list: dict[str, Any],
        source: str = "kubernetes:pods",
    ) -> KubernetesParseResult:
        """Parse a Kubernetes PodList API response."""
        result = KubernetesParseResult()
        collected_at = datetime.now(timezone.utc)

        for item in pod_list.get("items", []):
            try:
                self._parse_pod(item, collected_at, source, result)
            except Exception as exc:
                result.parse_errors.append(f"Pod parse error: {exc}")

        # Build pending queue aggregates per namespace/label-based service
        self._build_pending_queues(result, collected_at, source)

        return result

    def _parse_pod(
        self,
        item: dict[str, Any],
        collected_at: datetime,
        source: str,
        result: KubernetesParseResult,
    ) -> None:
        metadata = item.get("metadata", {})
        spec = item.get("spec", {})
        status = item.get("status", {})

        namespace = metadata.get("namespace", "default")
        if self.namespace_allowlist and namespace not in self.namespace_allowlist:
            return

        name = metadata.get("name", "unknown")
        labels = metadata.get("labels", {})
        annotations = metadata.get("annotations", {})
        workload_id = f"{namespace}/{name}"

        # Phase / status
        phase = status.get("phase", "Unknown")
        node_name = spec.get("nodeName")

        # GPU requests
        gpu_count = 0
        for container in spec.get("containers", []):
            resources = container.get("resources", {})
            gpu_count += self._parse_gpu_count(resources.get("requests", {}))
            gpu_count += self._parse_gpu_count(resources.get("limits", {}))

        if gpu_count == 0 and phase == "Pending":
            return

        # Workload type inference from labels
        workload_type = self._infer_workload_type(labels, annotations)

        # Communication intensity heuristic
        comm_intensity = self._infer_communication_intensity(gpu_count, labels)

        # SLA policy from annotations
        sla_policy_id = annotations.get("aurelius.io/sla-policy")
        priority_raw = labels.get("aurelius.io/priority", "0")
        try:
            priority_tier = int(priority_raw)
        except ValueError:
            priority_tier = 0

        # Migration allowed
        migration_allowed = annotations.get("aurelius.io/migration-allowed", "true").lower() != "false"

        # Current nodes (if Running/Succeeded)
        current_node_ids = [node_name] if node_name else []

        workload = WorkloadState(
            workload_id=workload_id,
            service_id=labels.get("app") or labels.get("serving.knative.dev/service"),
            workload_type=workload_type,
            priority_tier=priority_tier,
            current_region=None,
            current_node_ids=current_node_ids,
            current_gpu_ids=[],
            gpu_count_required=max(1, gpu_count),
            migration_allowed=migration_allowed,
            communication_intensity=comm_intensity,
            latency_sensitive=(
                labels.get("aurelius.io/latency-sensitive", "false").lower() == "true"
                or workload_type == WorkloadType.INFERENCE
            ),
            sla_policy_id=sla_policy_id,
            labels=labels,
        )
        result.workloads[workload_id] = workload

    def _build_pending_queues(
        self,
        result: KubernetesParseResult,
        collected_at: datetime,
        source: str,
    ) -> None:
        """Aggregate pending workloads into QueueState entries per namespace."""
        by_service: dict[str, list[WorkloadState]] = {}
        for w in result.workloads.values():
            svc = w.service_id or "default"
            by_service.setdefault(svc, []).append(w)

        for svc, workloads in by_service.items():
            pending = [w for w in workloads if not w.current_node_ids]
            if not pending:
                continue
            prov = Provenance(source=source, collected_at=collected_at, confidence=1.0)
            result.pending_queues[f"pending:{svc}"] = QueueState(
                queue_id=f"pending:{svc}",
                service_id=svc,
                pending_jobs=len(pending),
                queue_depth=len(pending),
                provenance=prov,
            )

    # ------------------------------------------------------------------
    # Resource parsing helpers
    # ------------------------------------------------------------------

    def _parse_gpu_count(self, resources: dict[str, Any]) -> int:
        for key in _GPU_RESOURCE_KEYS:
            if key in resources:
                try:
                    return int(resources[key])
                except (ValueError, TypeError):
                    pass
        return 0

    def _parse_cpu_millicores(self, cpu_str: Optional[str]) -> Optional[int]:
        if cpu_str is None:
            return None
        cpu_str = str(cpu_str).strip()
        if cpu_str.endswith("m"):
            try:
                return int(cpu_str[:-1])
            except ValueError:
                return None
        try:
            return int(float(cpu_str) * 1000)
        except ValueError:
            return None

    def _parse_bytes(self, mem_str: Optional[str]) -> Optional[int]:
        if mem_str is None:
            return None
        mem_str = str(mem_str).strip()
        suffix_map = {
            "Ki": 1024,
            "Mi": 1024 ** 2,
            "Gi": 1024 ** 3,
            "Ti": 1024 ** 4,
            "K": 1000,
            "M": 1000 ** 2,
            "G": 1000 ** 3,
        }
        for suffix, mult in suffix_map.items():
            if mem_str.endswith(suffix):
                try:
                    return int(mem_str[: -len(suffix)]) * mult
                except ValueError:
                    return None
        try:
            return int(mem_str)
        except ValueError:
            return None

    def _infer_workload_type(
        self, labels: dict[str, str], annotations: dict[str, str]
    ) -> WorkloadType:
        explicit = annotations.get("aurelius.io/workload-type") or labels.get("aurelius.io/workload-type")
        if explicit:
            try:
                return WorkloadType(explicit)
            except ValueError:
                pass
        app = labels.get("app", "").lower()
        if any(k in app for k in ("infer", "serve", "llm", "vllm", "triton")):
            return WorkloadType.INFERENCE
        if any(k in app for k in ("train", "finetune", "fine-tune")):
            return WorkloadType.FINE_TUNING
        if any(k in app for k in ("embed",)):
            return WorkloadType.EMBEDDING
        if any(k in app for k in ("batch",)):
            return WorkloadType.BATCH_TRAINING
        return WorkloadType.INFERENCE

    def _infer_communication_intensity(
        self, gpu_count: int, labels: dict[str, str]
    ) -> CommunicationIntensity:
        explicit = labels.get("aurelius.io/communication-intensity")
        if explicit:
            try:
                return CommunicationIntensity(explicit)
            except ValueError:
                pass
        if gpu_count >= 8:
            return CommunicationIntensity.HIGH
        if gpu_count >= 2:
            return CommunicationIntensity.MEDIUM
        return CommunicationIntensity.LOW
