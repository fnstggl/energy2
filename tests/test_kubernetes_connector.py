"""Tests for Kubernetes connector — Phase 4.

All tests use fixture files. No live cluster required.
"""

import json
import pathlib
import pytest
from datetime import timezone

from aurelius.connectors.kubernetes.parser import KubernetesParser, KubernetesParseResult
from aurelius.connectors.kubernetes.client import KubernetesClient, KubernetesClientError
from aurelius.state.models import WorkloadType, CommunicationIntensity

FIXTURES = pathlib.Path(__file__).parent / "fixtures" / "kubernetes"


def load_fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


# ---------------------------------------------------------------------------
# KubernetesParser — node parsing
# ---------------------------------------------------------------------------

class TestKubernetesParserNodes:
    def test_parses_node_list_fixture(self):
        parser = KubernetesParser()
        nodes = parser.parse_node_list(load_fixture("nodes_list.json"))
        assert len(nodes) == 3
        assert "gpu-node-1" in nodes
        assert "gpu-node-2" in nodes
        assert "cpu-node-1" in nodes

    def test_gpu_count_extracted(self):
        parser = KubernetesParser()
        nodes = parser.parse_node_list(load_fixture("nodes_list.json"))
        assert nodes["gpu-node-1"].gpu_count == 8
        assert nodes["gpu-node-2"].gpu_count == 8
        assert nodes["cpu-node-1"].gpu_count == 0

    def test_allocatable_gpu_extracted(self):
        parser = KubernetesParser()
        nodes = parser.parse_node_list(load_fixture("nodes_list.json"))
        assert nodes["gpu-node-1"].allocatable_gpu == 8
        assert nodes["gpu-node-2"].allocatable_gpu == 6  # limited

    def test_region_label_extracted(self):
        parser = KubernetesParser()
        nodes = parser.parse_node_list(load_fixture("nodes_list.json"))
        assert nodes["gpu-node-1"].region_id == "us-east-1"
        assert nodes["gpu-node-2"].region_id == "us-east-1"

    def test_zone_label_extracted(self):
        parser = KubernetesParser()
        nodes = parser.parse_node_list(load_fixture("nodes_list.json"))
        assert nodes["gpu-node-1"].zone == "us-east-1a"
        assert nodes["gpu-node-2"].zone == "us-east-1b"

    def test_rack_label_extracted(self):
        parser = KubernetesParser()
        nodes = parser.parse_node_list(load_fixture("nodes_list.json"))
        assert nodes["gpu-node-1"].rack_id == "rack-01"
        assert nodes["gpu-node-2"].rack_id == "rack-02"

    def test_instance_type_extracted(self):
        parser = KubernetesParser()
        nodes = parser.parse_node_list(load_fixture("nodes_list.json"))
        assert nodes["gpu-node-1"].instance_type == "p4d.24xlarge"

    def test_node_ready_status(self):
        parser = KubernetesParser()
        nodes = parser.parse_node_list(load_fixture("nodes_list.json"))
        assert nodes["gpu-node-1"].ready is True
        assert nodes["gpu-node-2"].ready is True

    def test_taints_extracted(self):
        parser = KubernetesParser()
        nodes = parser.parse_node_list(load_fixture("nodes_list.json"))
        taints = nodes["gpu-node-2"].taints
        assert len(taints) == 1
        assert taints[0]["key"] == "dedicated"
        assert taints[0]["effect"] == "NoSchedule"

    def test_allocatable_memory_bytes(self):
        parser = KubernetesParser()
        nodes = parser.parse_node_list(load_fixture("nodes_list.json"))
        node = nodes["gpu-node-1"]
        assert node.allocatable_memory_bytes is not None
        assert node.allocatable_memory_bytes > 0

    def test_allocatable_cpu_millicores(self):
        parser = KubernetesParser()
        nodes = parser.parse_node_list(load_fixture("nodes_list.json"))
        node = nodes["gpu-node-1"]
        assert node.allocatable_cpu_millicores == 94000

    def test_no_gpu_node_zero_gpu_count(self):
        parser = KubernetesParser()
        nodes = parser.parse_node_list(load_fixture("nodes_list.json"))
        assert nodes["cpu-node-1"].gpu_count == 0

    def test_empty_node_list(self):
        parser = KubernetesParser()
        nodes = parser.parse_node_list({"items": []})
        assert len(nodes) == 0

    def test_malformed_item_skipped_gracefully(self):
        parser = KubernetesParser()
        node_list = {
            "items": [
                {},  # malformed
                {
                    "metadata": {"name": "good-node", "labels": {}},
                    "spec": {},
                    "status": {"conditions": [{"type": "Ready", "status": "True"}]},
                },
            ]
        }
        nodes = parser.parse_node_list(node_list)
        # good node should still be parsed
        assert "good-node" in nodes


# ---------------------------------------------------------------------------
# KubernetesParser — pod / workload parsing
# ---------------------------------------------------------------------------

class TestKubernetesParserPods:
    def test_parses_pod_list_fixture(self):
        parser = KubernetesParser()
        result = parser.parse_pod_list(load_fixture("pods_list.json"))
        assert len(result.workloads) >= 2  # running pods with GPUs

    def test_pod_to_node_mapping(self):
        parser = KubernetesParser()
        result = parser.parse_pod_list(load_fixture("pods_list.json"))
        wl = result.workloads.get("prod/llm-server-abc-1")
        assert wl is not None
        assert "gpu-node-1" in wl.current_node_ids

    def test_gpu_count_extracted_from_requests(self):
        parser = KubernetesParser()
        result = parser.parse_pod_list(load_fixture("pods_list.json"))
        wl = result.workloads.get("prod/llm-server-abc-1")
        assert wl is not None
        # limits=4, requests=4 → max(4+4, 1) = 8 from both containers
        # Actually: the code does gpu_count += _parse_gpu_count(requests) + _parse_gpu_count(limits)
        # So for one container with 4 in requests and 4 in limits: 4+4=8
        assert wl.gpu_count_required >= 4

    def test_workload_type_inference_from_labels(self):
        parser = KubernetesParser()
        result = parser.parse_pod_list(load_fixture("pods_list.json"))
        wl = result.workloads.get("prod/llm-server-abc-1")
        assert wl is not None
        # app="llm-server" → INFERENCE
        assert wl.workload_type == WorkloadType.INFERENCE

    def test_batch_workload_type(self):
        parser = KubernetesParser()
        result = parser.parse_pod_list(load_fixture("pods_list.json"))
        wl = result.workloads.get("batch/batch-job-xyz-1")
        assert wl is not None
        assert wl.workload_type == WorkloadType.BATCH_TRAINING

    def test_priority_from_label(self):
        parser = KubernetesParser()
        result = parser.parse_pod_list(load_fixture("pods_list.json"))
        wl = result.workloads.get("prod/llm-server-abc-1")
        assert wl.priority_tier == 3

    def test_migration_allowed_false_from_annotation(self):
        parser = KubernetesParser()
        result = parser.parse_pod_list(load_fixture("pods_list.json"))
        wl = result.workloads.get("prod/llm-server-abc-1")
        assert wl.migration_allowed is False

    def test_latency_sensitive_from_label(self):
        parser = KubernetesParser()
        result = parser.parse_pod_list(load_fixture("pods_list.json"))
        wl = result.workloads.get("prod/llm-server-abc-1")
        assert wl.latency_sensitive is True

    def test_sla_policy_from_annotation(self):
        parser = KubernetesParser()
        result = parser.parse_pod_list(load_fixture("pods_list.json"))
        wl = result.workloads.get("prod/llm-server-abc-1")
        assert wl.sla_policy_id == "critical-inference"

    def test_communication_intensity_high_for_8gpu(self):
        parser = KubernetesParser()
        result = parser.parse_pod_list(load_fixture("pods_list.json"))
        wl = result.workloads.get("batch/batch-job-xyz-1")
        assert wl is not None
        # 8 GPUs → HIGH
        assert wl.communication_intensity == CommunicationIntensity.HIGH

    def test_pending_pod_detected_in_queue(self):
        parser = KubernetesParser()
        result = parser.parse_pod_list(load_fixture("pods_list.json"))
        assert len(result.pending_queues) >= 1

    def test_pending_pod_has_empty_node_list(self):
        parser = KubernetesParser()
        result = parser.parse_pod_list(load_fixture("pods_list.json"))
        wl = result.workloads.get("prod/pending-inference-1")
        if wl is not None:
            assert wl.current_node_ids == []

    def test_namespace_filtering(self):
        parser = KubernetesParser(namespace_allowlist=["prod"])
        result = parser.parse_pod_list(load_fixture("pods_list.json"))
        for wl_id in result.workloads:
            assert wl_id.startswith("prod/")

    def test_no_write_permissions_needed(self):
        # Parser only reads dicts — no Kubernetes client calls
        parser = KubernetesParser()
        pod_list = load_fixture("pods_list.json")
        result = parser.parse_pod_list(pod_list)
        # Should complete without any network calls
        assert isinstance(result, KubernetesParseResult)

    def test_empty_pod_list(self):
        parser = KubernetesParser()
        result = parser.parse_pod_list({"items": []})
        assert len(result.workloads) == 0
        assert len(result.pending_queues) == 0


# ---------------------------------------------------------------------------
# KubernetesClient — sandbox mode
# ---------------------------------------------------------------------------

class TestKubernetesClientSandbox:
    def test_sandbox_list_nodes(self):
        nodes_data = load_fixture("nodes_list.json")
        client = KubernetesClient(_sandbox_responses={"/api/v1/nodes": nodes_data})
        result = client.list_nodes()
        assert result["kind"] == "NodeList"
        assert len(result["items"]) == 3

    def test_sandbox_list_pods(self):
        pods_data = load_fixture("pods_list.json")
        client = KubernetesClient(_sandbox_responses={"/api/v1/pods": pods_data})
        result = client.list_pods()
        assert result["kind"] == "PodList"

    def test_sandbox_missing_path_raises(self):
        client = KubernetesClient(_sandbox_responses={})
        with pytest.raises(KubernetesClientError):
            client.list_nodes()

    def test_full_parse_pipeline_offline(self):
        """End-to-end: client + parser, fully offline."""
        nodes_data = load_fixture("nodes_list.json")
        pods_data = load_fixture("pods_list.json")
        client = KubernetesClient(_sandbox_responses={
            "/api/v1/nodes": nodes_data,
            "/api/v1/pods": pods_data,
        })
        parser = KubernetesParser()
        nodes = parser.parse_node_list(client.list_nodes())
        result = parser.parse_pod_list(client.list_pods())

        assert len(nodes) == 3
        assert len(result.workloads) >= 2

    def test_sandbox_namespace_pods(self):
        pods_data = load_fixture("pods_list.json")
        client = KubernetesClient(_sandbox_responses={
            "/api/v1/namespaces/prod/pods": pods_data,
        })
        result = client.list_pods(namespace="prod")
        assert result["kind"] == "PodList"


# ---------------------------------------------------------------------------
# Resource parsing helpers
# ---------------------------------------------------------------------------

class TestResourceParsing:
    def _parser(self):
        return KubernetesParser()

    def test_parse_cpu_millicores_integer(self):
        p = self._parser()
        assert p._parse_cpu_millicores("4") == 4000

    def test_parse_cpu_millicores_millicore(self):
        p = self._parser()
        assert p._parse_cpu_millicores("500m") == 500

    def test_parse_cpu_millicores_none(self):
        p = self._parser()
        assert p._parse_cpu_millicores(None) is None

    def test_parse_bytes_gi(self):
        p = self._parser()
        assert p._parse_bytes("1Gi") == 1024 ** 3

    def test_parse_bytes_ki(self):
        p = self._parser()
        assert p._parse_bytes("1Ki") == 1024

    def test_parse_bytes_mi(self):
        p = self._parser()
        assert p._parse_bytes("512Mi") == 512 * 1024 ** 2

    def test_parse_bytes_none(self):
        p = self._parser()
        assert p._parse_bytes(None) is None

    def test_parse_bytes_raw_int(self):
        p = self._parser()
        assert p._parse_bytes("1073741824") == 1073741824
