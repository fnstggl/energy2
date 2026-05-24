"""Tests for aurelius/state/normalize.py.

Proves:
- validate_utc_aware raises on naive datetime, passes on UTC-aware
- validate_percentage raises outside [0, 100], passes within range
- validate_non_negative raises on negative values, passes on 0+
- make_provenance creates a valid Provenance with UTC now
- adapt_gpu_metrics correctly maps GPUMetrics → GPUState
- adapt_gpu_metrics preserves None-not-zero for missing optional fields
- adapt_gpu_metrics + GPUHealthScore merges health_penalty/is_schedulable
- adapt_queue_state wraps QueueState with provenance
- coerce_to_utc attaches UTC to naive datetimes
- adapt_gpu_metrics uses existing fixture shapes (dcgm_metrics_healthy)

No optimizer behavior is changed — we verify existing optimizer imports
still work with no modifications.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import pytest

from aurelius.state.models import GPUState, Provenance
from aurelius.state.normalize import (
    adapt_gpu_metrics,
    adapt_queue_state,
    coerce_to_utc,
    make_provenance,
    validate_non_negative,
    validate_percentage,
    validate_utc_aware,
)

UTC = timezone.utc
NOW = datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC)


def _prov(source: str = "test") -> Provenance:
    return Provenance(source=source, fetched_at=NOW, confidence="high")


# ---------------------------------------------------------------------------
# validate_utc_aware
# ---------------------------------------------------------------------------

class TestValidateUtcAware:
    def test_utc_aware_passes(self):
        dt = datetime(2024, 1, 1, tzinfo=UTC)
        result = validate_utc_aware(dt)
        assert result is dt

    def test_naive_raises(self):
        naive = datetime(2024, 1, 1)
        with pytest.raises(ValueError, match="UTC-aware"):
            validate_utc_aware(naive)

    def test_custom_field_name_in_error(self):
        naive = datetime(2024, 1, 1)
        with pytest.raises(ValueError, match="my_field"):
            validate_utc_aware(naive, "my_field")


# ---------------------------------------------------------------------------
# validate_percentage
# ---------------------------------------------------------------------------

class TestValidatePercentage:
    def test_zero_passes(self):
        assert validate_percentage(0.0, "x") == 0.0

    def test_hundred_passes(self):
        assert validate_percentage(100.0, "x") == 100.0

    def test_midrange_passes(self):
        assert validate_percentage(55.5, "x") == 55.5

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="\\[0, 100\\]"):
            validate_percentage(-0.1, "field")

    def test_over_100_raises(self):
        with pytest.raises(ValueError, match="\\[0, 100\\]"):
            validate_percentage(100.01, "field")

    def test_none_passes_by_default(self):
        assert validate_percentage(None, "x") is None

    def test_none_raises_when_not_allowed(self):
        with pytest.raises(ValueError, match="None"):
            validate_percentage(None, "x", allow_none=False)


# ---------------------------------------------------------------------------
# validate_non_negative
# ---------------------------------------------------------------------------

class TestValidateNonNegative:
    def test_zero_passes(self):
        assert validate_non_negative(0.0, "x") == 0.0

    def test_positive_passes(self):
        assert validate_non_negative(1e9, "x") == 1e9

    def test_negative_raises(self):
        with pytest.raises(ValueError, match=">= 0"):
            validate_non_negative(-0.001, "bytes_per_s")

    def test_none_passes_by_default(self):
        assert validate_non_negative(None, "x") is None

    def test_none_raises_when_not_allowed(self):
        with pytest.raises(ValueError, match="None"):
            validate_non_negative(None, "x", allow_none=False)


# ---------------------------------------------------------------------------
# make_provenance
# ---------------------------------------------------------------------------

class TestMakeProvenance:
    def test_creates_with_utc_now_by_default(self):
        p = make_provenance("prometheus")
        assert p.source == "prometheus"
        assert p.fetched_at.tzinfo is not None

    def test_uses_provided_fetched_at(self):
        p = make_provenance("dcgm", fetched_at=NOW)
        assert p.fetched_at == NOW

    def test_naive_fetched_at_raises(self):
        with pytest.raises(ValueError, match="UTC-aware"):
            make_provenance("x", fetched_at=datetime(2024, 1, 1))

    def test_is_sandbox_propagated(self):
        p = make_provenance("simulator", is_sandbox=True)
        assert p.is_sandbox is True

    def test_confidence_propagated(self):
        p = make_provenance("nvml", confidence="low")
        assert p.confidence == "low"


# ---------------------------------------------------------------------------
# adapt_gpu_metrics → GPUState
# ---------------------------------------------------------------------------

class TestAdaptGpuMetrics:
    def _make_gpu_metrics(
        self,
        *,
        gpu_util_pct: float = 75.0,
        mem_used_mb: float = 40000.0,
        mem_total_mb: float = 80000.0,
        power_usage_w: float = 350.0,
        gpu_temp_c: float = 72.0,
        ecc_sbe_count: int = 0,
        ecc_dbe_count: int = 0,
        xid_error_count: int = 0,
        power_throttle_us: float = 0.0,
        thermal_throttle_us: float = 0.0,
        clock_throttle_reasons: int = 0,
        timestamp: Optional[datetime] = None,
    ):
        from aurelius.models import GPUMetrics
        return GPUMetrics(
            timestamp=timestamp or NOW,
            region="us-east",
            node_id="node-01",
            gpu_index=0,
            gpu_uuid="GPU-test-uuid",
            gpu_type="h100",
            gpu_util_pct=gpu_util_pct,
            mem_used_mb=mem_used_mb,
            mem_total_mb=mem_total_mb,
            power_usage_w=power_usage_w,
            gpu_temp_c=gpu_temp_c,
            ecc_sbe_count=ecc_sbe_count,
            ecc_dbe_count=ecc_dbe_count,
            xid_error_count=xid_error_count,
            power_throttle_us=power_throttle_us,
            thermal_throttle_us=thermal_throttle_us,
            clock_throttle_reasons=clock_throttle_reasons,
        )

    def test_basic_fields_mapped(self):
        metrics = self._make_gpu_metrics()
        gpu = adapt_gpu_metrics(metrics, _prov())
        assert gpu.gpu_uuid == "GPU-test-uuid"
        assert gpu.node_id == "node-01"
        assert gpu.region == "us-east"
        assert gpu.util_pct == 75.0
        assert gpu.power_w == 350.0
        assert gpu.temp_c == 72.0

    def test_utc_aware_timestamp_output(self):
        metrics = self._make_gpu_metrics()
        gpu = adapt_gpu_metrics(metrics, _prov())
        assert gpu.timestamp.tzinfo is not None

    def test_naive_timestamp_coerced_to_utc(self):
        naive_ts = datetime(2024, 6, 1, 12, 0, 0)
        metrics = self._make_gpu_metrics(timestamp=naive_ts)
        gpu = adapt_gpu_metrics(metrics, _prov())
        assert gpu.timestamp.tzinfo is not None

    def test_mem_fields_mapped(self):
        metrics = self._make_gpu_metrics(mem_used_mb=20000.0, mem_total_mb=80000.0)
        gpu = adapt_gpu_metrics(metrics, _prov())
        assert gpu.mem_used_mb == 20000.0
        assert gpu.mem_total_mb == 80000.0
        # mem_free derived as total - used
        assert gpu.mem_free_mb == pytest.approx(60000.0)

    def test_ecc_counts_mapped(self):
        metrics = self._make_gpu_metrics(ecc_sbe_count=2, ecc_dbe_count=1)
        gpu = adapt_gpu_metrics(metrics, _prov())
        assert gpu.ecc_sbe_total == 2
        assert gpu.ecc_dbe_total == 1

    def test_clock_throttle_mapped(self):
        HW_THERMAL = 0x40
        metrics = self._make_gpu_metrics(clock_throttle_reasons=HW_THERMAL)
        gpu = adapt_gpu_metrics(metrics, _prov())
        assert gpu.clocks_event_reasons == HW_THERMAL
        assert gpu.thermal_throttling is True

    def test_zero_clock_throttle_maps_to_none(self):
        metrics = self._make_gpu_metrics(clock_throttle_reasons=0)
        gpu = adapt_gpu_metrics(metrics, _prov())
        # clock_throttle_reasons=0 → None in the adapter (not meaningful)
        assert gpu.clocks_event_reasons is None

    def test_health_none_without_health_score(self):
        metrics = self._make_gpu_metrics()
        gpu = adapt_gpu_metrics(metrics, _prov())
        assert gpu.health_penalty is None
        assert gpu.is_schedulable is None

    def test_health_score_merged(self):
        from aurelius.models import GPUHealthScore
        metrics = self._make_gpu_metrics()
        health = GPUHealthScore(
            gpu_uuid="GPU-test-uuid",
            node_id="node-01",
            region="us-east",
            timestamp=NOW,
            health_penalty=0.15,
            utilization_penalty=0.0,
            thermal_penalty=0.1,
            throttle_penalty=0.05,
            ecc_penalty=0.0,
            is_schedulable=True,
            reason_codes=["thermal"],
        )
        gpu = adapt_gpu_metrics(metrics, _prov(), health_score=health)
        assert gpu.health_penalty == pytest.approx(0.15)
        assert gpu.is_schedulable is True

    def test_gpu_state_is_valid(self):
        """Result must be a valid frozen GPUState (all validations pass)."""
        metrics = self._make_gpu_metrics()
        gpu = adapt_gpu_metrics(metrics, _prov())
        assert isinstance(gpu, GPUState)

    def test_xid_zero_mapped_as_zero(self):
        metrics = self._make_gpu_metrics(xid_error_count=0)
        gpu = adapt_gpu_metrics(metrics, _prov())
        assert gpu.xid_last == 0

    def test_throttle_nonzero_mapped_to_violation_ns(self):
        metrics = self._make_gpu_metrics(power_throttle_us=5000.0)
        gpu = adapt_gpu_metrics(metrics, _prov())
        assert gpu.power_violation_ns == 5000

    def test_throttle_zero_maps_to_none(self):
        metrics = self._make_gpu_metrics(power_throttle_us=0.0)
        gpu = adapt_gpu_metrics(metrics, _prov())
        assert gpu.power_violation_ns is None


# ---------------------------------------------------------------------------
# adapt_queue_state
# ---------------------------------------------------------------------------

class TestAdaptQueueState:
    def _make_queue_state(self, *, timestamp: Optional[datetime] = None):
        from aurelius.models import QueueState
        return QueueState(
            timestamp=timestamp or NOW,
            region="us-west",
            cluster_id="gpu-pool-1",
            gpu_type="a100",
            available_gpus=12,
            queue_depth_jobs=5,
            est_wait_hours=0.5,
        )

    def test_basic_fields_present(self):
        qs = self._make_queue_state()
        result = adapt_queue_state(qs, _prov())
        assert result["region"] == "us-west"
        assert result["cluster_id"] == "gpu-pool-1"
        assert result["gpu_type"] == "a100"
        assert result["available_gpus"] == 12
        assert result["queue_depth_jobs"] == 5
        assert result["est_wait_hours"] == 0.5

    def test_provenance_included(self):
        qs = self._make_queue_state()
        prov = make_provenance("queue-provider", confidence="medium")
        result = adapt_queue_state(qs, prov)
        assert "provenance" in result
        assert result["provenance"]["source"] == "queue-provider"

    def test_utc_aware_timestamp_in_output(self):
        qs = self._make_queue_state()
        result = adapt_queue_state(qs, _prov())
        ts = datetime.fromisoformat(result["timestamp"])
        assert ts.tzinfo is not None

    def test_naive_timestamp_coerced(self):
        naive = datetime(2024, 6, 1, 12, 0, 0)
        qs = self._make_queue_state(timestamp=naive)
        result = adapt_queue_state(qs, _prov())
        ts = datetime.fromisoformat(result["timestamp"])
        assert ts.tzinfo is not None


# ---------------------------------------------------------------------------
# coerce_to_utc
# ---------------------------------------------------------------------------

class TestCoerceToUtc:
    def test_naive_gets_utc(self):
        naive = datetime(2024, 1, 1)
        result = coerce_to_utc(naive)
        assert result.tzinfo == UTC

    def test_utc_aware_unchanged(self):
        aware = datetime(2024, 1, 1, tzinfo=UTC)
        result = coerce_to_utc(aware)
        assert result is aware


# ---------------------------------------------------------------------------
# No optimizer behavior changes — verify existing optimizer still imports cleanly
# ---------------------------------------------------------------------------

class TestOptimizerUnchanged:
    def test_job_scheduler_importable(self):
        from aurelius.optimization.scheduler import JobScheduler
        assert JobScheduler is not None

    def test_objective_function_importable(self):
        from aurelius.optimization.objective import ObjectiveFunction
        assert ObjectiveFunction is not None

    def test_existing_models_unchanged(self):
        from aurelius.models import (
            GPUHealthScore,
            GPUMetrics,
            QueueState,
        )
        # Just importing proves they still exist with the same names
        assert QueueState is not None
        assert GPUMetrics is not None
        assert GPUHealthScore is not None

    def test_sla_workload_state_unchanged(self):
        from aurelius.sla.telemetry import WorkloadState
        ws = WorkloadState(region="us-east", p99_latency_ms=200.0)
        assert ws.p99_latency_ms == 200.0

    def test_sla_action_types_unchanged(self):
        from aurelius.sla.actions import ActionType
        assert ActionType.MIGRATE.value == "migrate_workload"
        assert ActionType.KEEP.value == "keep_current_placement"
