"""Generate fake Kubernetes API V1NodeList and V1PodList payloads.

Outputs match the exact JSON shape returned by the Kubernetes API server.
The production KubernetesConnector parses these without modification,
verifying that simulator and real K8s share the same parsing path.
"""

from __future__ import annotations

from typing import Any

from ..model import SimCluster, SimNode, SimWorkload


def generate_node_list(cluster: SimCluster) -> dict[str, Any]:
    """Generate a fake V1NodeList payload from simulator cluster state."""
    items = []
    for region in cluster.regions.values():
        for node in region.nodes:
            items.append(_build_node_object(node))

    return {
        "apiVersion": "v1",
        "kind": "NodeList",
        "metadata": {"resourceVersion": str(cluster.tick * 100)},
        "items": items,
    }


def _build_node_object(node: SimNode) -> dict[str, Any]:
    gpu_count = node.gpu_count
    gpu_type = node.labels.get("gpu-type", "nvidia.com/gpu")

    labels = {
        "topology.kubernetes.io/region": node.region_id,
        "topology.kubernetes.io/zone": node.zone,
        "kubernetes.io/hostname": node.node_id,
        "node.kubernetes.io/instance-type": node.instance_type,
        "nvidia.com/gpu.product": gpu_type.upper().replace("-", "_"),
        "nvidia.com/gpu.count": str(gpu_count),
    }
    labels.update(node.labels)

    # Annotate allocated GPU count
    allocated = node.gpu_allocated_count

    return {
        "apiVersion": "v1",
        "kind": "Node",
        "metadata": {
            "name": node.node_id,
            "labels": labels,
            "annotations": {
                "aurelius.io/rack-id": node.rack_id,
                "aurelius.io/gpu-allocated": str(allocated),
            },
        },
        "spec": {
            "taints": node.taints,
        },
        "status": {
            "capacity": {
                "cpu": "96",
                "memory": "768Gi",
                "nvidia.com/gpu": str(gpu_count),
                "ephemeral-storage": "1Ti",
            },
            "allocatable": {
                "cpu": "94",
                "memory": "750Gi",
                "nvidia.com/gpu": str(gpu_count),
                "ephemeral-storage": "900Gi",
            },
            "conditions": [
                {
                    "type": "Ready",
                    "status": "True",
                    "reason": "KubeletReady",
                    "message": "kubelet is posting ready status",
                },
                {
                    "type": "MemoryPressure",
                    "status": "False",
                },
                {
                    "type": "DiskPressure",
                    "status": "False",
                },
                {
                    "type": "PIDPressure",
                    "status": "False",
                },
            ],
            "nodeInfo": {
                "machineID": node.node_id + "-machine",
                "systemUUID": node.node_id + "-uuid",
                "kernelVersion": "5.15.0-91-generic",
                "osImage": "Ubuntu 22.04.3 LTS",
                "containerRuntimeVersion": "containerd://1.6.24",
                "kubeletVersion": "v1.28.3",
                "kubeProxyVersion": "v1.28.3",
                "architecture": "amd64",
                "operatingSystem": "linux",
            },
        },
    }


def generate_pod_list(cluster: SimCluster) -> dict[str, Any]:
    """Generate a fake V1PodList payload from simulator cluster state."""
    items = []

    for workload in cluster.workloads.values():
        for pod in _build_pod_objects(workload, cluster):
            items.append(pod)

    # Add some pending pods to simulate queue pressure
    for region in cluster.regions.values():
        for queue in region.queues:
            # proxy: 1 pending pod per 100 queued requests
            pending = min(queue.queue_depth // 100, 5)
            for i in range(pending):
                items.append(_build_pending_pod(queue.service_id, region.region_id, i))

    return {
        "apiVersion": "v1",
        "kind": "PodList",
        "metadata": {"resourceVersion": str(cluster.tick * 100 + 1)},
        "items": items,
    }


def _build_pod_objects(workload: SimWorkload, cluster: SimCluster) -> list[dict[str, Any]]:
    """Build one pod per GPU assigned to the workload."""
    pods = []
    if not workload.node_ids:
        return pods

    # Build one pod representing the whole workload (multi-GPU pod)
    node_id = workload.node_ids[0] if workload.node_ids else None
    if node_id is None:
        return pods

    gpu_request = str(workload.gpu_count_required)
    mem_mb = workload.memory_required_bytes // (1024 * 1024)

    pod = {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": f"{workload.service_id}-0",
            "namespace": "default",
            "labels": {
                "app": workload.service_id,
                "workload-type": workload.workload_type,
                "priority": workload.priority_tier,
            },
        },
        "spec": {
            "nodeName": node_id,
            "containers": [
                {
                    "name": workload.service_id,
                    "image": f"aurelius-sim/{workload.workload_type}:latest",
                    "resources": {
                        "requests": {
                            "nvidia.com/gpu": gpu_request,
                            "memory": f"{mem_mb}Mi",
                            "cpu": "8",
                        },
                        "limits": {
                            "nvidia.com/gpu": gpu_request,
                            "memory": f"{mem_mb}Mi",
                            "cpu": "16",
                        },
                    },
                }
            ],
            "tolerations": [],
        },
        "status": {
            "phase": "Running",
            "conditions": [
                {"type": "Ready", "status": "True"},
                {"type": "PodScheduled", "status": "True"},
            ],
            "startTime": "2024-01-01T00:00:00Z",
            "containerStatuses": [
                {
                    "name": workload.service_id,
                    "ready": True,
                    "restartCount": 0,
                    "state": {"running": {"startedAt": "2024-01-01T00:00:00Z"}},
                }
            ],
        },
    }
    pods.append(pod)
    return pods


def _build_pending_pod(service_id: str, region_id: str, index: int) -> dict[str, Any]:
    """Build a pending pod representing queued demand."""
    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": f"{service_id}-pending-{index}",
            "namespace": "default",
            "labels": {
                "app": service_id,
                "status": "pending",
            },
        },
        "spec": {
            "containers": [
                {
                    "name": service_id,
                    "image": "aurelius-sim/inference:latest",
                    "resources": {
                        "requests": {
                            "nvidia.com/gpu": "1",
                            "cpu": "4",
                        },
                        "limits": {
                            "nvidia.com/gpu": "1",
                            "cpu": "8",
                        },
                    },
                }
            ],
            "nodeSelector": {
                "topology.kubernetes.io/region": region_id,
            },
        },
        "status": {
            "phase": "Pending",
            "conditions": [
                {
                    "type": "PodScheduled",
                    "status": "False",
                    "reason": "Unschedulable",
                    "message": "0/4 nodes are available: insufficient nvidia.com/gpu.",
                }
            ],
        },
    }
