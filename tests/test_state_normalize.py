"""Tests for aurelius/state/normalize.py — normalization utilities."""

import pytest
from datetime import datetime, timezone

from aurelius.state.normalize import (
    ensure_utc,
    validate_percentage,
    validate_non_negative,
    normalize_gpu_state,
    normalize_inference_service,
    normalize_queue_state,
)
from aurelius.state.models import Provenance

UTC = timezone.utc
NOW = datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC)
PROV = Provenance(source="test", collected_at=NOW)


class TestEnsureUtc:
    def test_naive_gets_utc(self):
        naive = datetime(2025, 1, 1, 12, 0, 0)
        aware = ensure_utc(naive)
        assert aware.tzinfo == UTC

    def test_aware_passes_through(self):
        aware = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
        result = ensure_utc(aware)
        assert result is aware


class TestValidatePercentage:
    def test_none_passes(self):
        assert validate_percentage(None) is None

    def test_valid_range(self):
        assert validate_percentage(0.0) == 0.0
        assert validate_percentage(100.0) == 100.0
        assert validate_percentage(50.5) == 50.5

    def test_above_100_raises(self):
        with pytest.raises(ValueError):
            validate_percentage(100.1)

    def test_below_0_raises(self):
        with pytest.raises(ValueError):
            validate_percentage(-0.001)

    def test_field_name_in_error(self):
        with pytest.raises(ValueError, match="utilization"):
            validate_percentage(200.0, "utilization")


class TestValidateNonNegative:
    def test_none_passes(self):
        assert validate_non_negative(None) is None

    def test_zero_passes(self):
        assert validate_non_negative(0.0) == 0.0

    def test_positive_passes(self):
        assert validate_non_negative(1.5) == 1.5

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            validate_non_negative(-0.001)


class TestNormalizeGPUState:
    def test_minimal_required(self):
        gpu = normalize_gpu_state({"gpu_id": "G1"}, PROV)
        assert gpu.gpu_id == "G1"
        assert gpu.utilization_pct is None
        assert gpu.power_watts is None
        assert gpu.temperature_c is None

    def test_all_optional_none_when_missing(self):
        gpu = normalize_gpu_state({"gpu_id": "G1"}, PROV)
        assert gpu.uuid is None
        assert gpu.node_id is None
        assert gpu.xid_error_count is None
        assert gpu.nvlink_rx_bytes_per_sec is None

    def test_none_not_zero(self):
        """Critical invariant: missing metrics must be None, never 0."""
        gpu = normalize_gpu_state({"gpu_id": "G1"}, PROV)
        assert gpu.utilization_pct is None
        assert gpu.memory_used_bytes is None
        assert gpu.pcie_rx_bytes_per_sec is None

    def test_unparseable_float_becomes_none(self):
        gpu = normalize_gpu_state({"gpu_id": "G1", "utilization_pct": "bad"}, PROV)
        assert gpu.utilization_pct is None

    def test_valid_full_gpu(self):
        raw = {
            "gpu_id": "GPU-abc",
            "uuid": "GPU-abc",
            "node_id": "node-1",
            "model": "H100",
            "utilization_pct": 75.5,
            "sm_activity_pct": 72.0,
            "memory_used_bytes": 42949672960,
            "memory_total_bytes": 85899345920,
            "power_watts": 380.0,
            "temperature_c": 72.0,
            "thermal_throttle_active": False,
            "xid_error_count": 0,
            "nvlink_rx_bytes_per_sec": 1234567.0,
        }
        gpu = normalize_gpu_state(raw, PROV)
        assert gpu.utilization_pct == 75.5
        assert gpu.power_watts == 380.0
        assert gpu.temperature_c == 72.0
        assert gpu.xid_error_count == 0
        assert gpu.nvlink_rx_bytes_per_sec == 1234567.0

    def test_out_of_range_pct_silently_becomes_none(self):
        gpu = normalize_gpu_state({"gpu_id": "G1", "utilization_pct": 150.0}, PROV)
        assert gpu.utilization_pct is None

    def test_thermal_throttle_from_int(self):
        gpu = normalize_gpu_state({"gpu_id": "G1", "thermal_throttle_active": 1}, PROV)
        assert gpu.thermal_throttle_active is True

    def test_thermal_throttle_from_string(self):
        gpu = normalize_gpu_state({"gpu_id": "G1", "thermal_throttle_active": "true"}, PROV)
        assert gpu.thermal_throttle_active is True

    def test_provenance_attached(self):
        gpu = normalize_gpu_state({"gpu_id": "G1"}, PROV)
        assert gpu.provenance is PROV

    def test_assigned_workload_ids(self):
        gpu = normalize_gpu_state(
            {"gpu_id": "G1", "assigned_workload_ids": ["w1", "w2"]},
            PROV,
        )
        assert gpu.assigned_workload_ids == ["w1", "w2"]


class TestNormalizeInferenceService:
    def test_minimal_required(self):
        svc = normalize_inference_service({"service_id": "svc-1"}, PROV)
        assert svc.service_id == "svc-1"
        assert svc.requests_per_second is None
        assert svc.kv_cache_usage_pct is None

    def test_none_not_zero(self):
        svc = normalize_inference_service({"service_id": "svc-1"}, PROV)
        assert svc.ttft_p50_ms is None
        assert svc.tokens_per_second is None
        assert svc.prefix_cache_hit_rate_pct is None

    def test_runtime_normalization(self):
        svc = normalize_inference_service({"service_id": "s", "runtime": "vllm"}, PROV)
        from aurelius.state.models import RuntimeType
        assert svc.runtime == RuntimeType.VLLM

    def test_unknown_runtime_becomes_unknown(self):
        svc = normalize_inference_service({"service_id": "s", "runtime": "unknown_runtime"}, PROV)
        from aurelius.state.models import RuntimeType
        assert svc.runtime == RuntimeType.UNKNOWN

    def test_pct_out_of_range_becomes_none(self):
        svc = normalize_inference_service({"service_id": "s", "kv_cache_usage_pct": 150.0}, PROV)
        assert svc.kv_cache_usage_pct is None

    def test_full_vllm_service(self):
        raw = {
            "service_id": "llm-server",
            "runtime": "vllm",
            "requests_per_second": 42.5,
            "tokens_per_second": 3200.0,
            "ttft_p50_ms": 80.0,
            "ttft_p95_ms": 180.0,
            "ttft_p99_ms": 320.0,
            "queue_depth": 12,
            "active_sequences": 8,
            "kv_cache_usage_pct": 68.0,
            "prefix_cache_hit_rate_pct": 45.0,
        }
        svc = normalize_inference_service(raw, PROV)
        assert svc.requests_per_second == 42.5
        assert svc.ttft_p99_ms == 320.0
        assert svc.kv_cache_usage_pct == 68.0


class TestNormalizeQueueState:
    def test_minimal(self):
        q = normalize_queue_state({"queue_id": "q1"}, PROV)
        assert q.queue_id == "q1"
        assert q.pending_jobs is None
        assert q.p95_wait_ms is None

    def test_none_not_zero(self):
        q = normalize_queue_state({"queue_id": "q1"}, PROV)
        assert q.queue_depth is None
        assert q.arrival_rate_per_sec is None

    def test_full_queue(self):
        raw = {
            "queue_id": "q1",
            "service_id": "svc-1",
            "pending_jobs": 10,
            "queue_depth": 10,
            "oldest_pending_age_sec": 30.0,
            "p95_wait_ms": 450.0,
            "arrival_rate_per_sec": 45.0,
            "service_rate_per_sec": 42.0,
        }
        q = normalize_queue_state(raw, PROV)
        assert q.pending_jobs == 10
        assert q.p95_wait_ms == 450.0

    def test_unparseable_value_becomes_none(self):
        q = normalize_queue_state({"queue_id": "q1", "pending_jobs": "N/A"}, PROV)
        assert q.pending_jobs is None
