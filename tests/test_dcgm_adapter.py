"""Tests for Phase 3: DCGM adapter.

Proves:
- DCGM fixture parses into GPUState objects correctly
- All GPU UUIDs are discovered from snapshot
- Label-filtered field extraction works correctly
- Missing optional metrics produce None in GPUState (not 0)
- health_penalty is derived correctly
- thermal_throttling property reflects clocks_event_reasons bitmask
- mem_total_mb is derived from used + free + reserved
- Node-level node_id is extracted from labels when possible
- Adapter generates unknown_metrics list
- thermal_violation_ns is nanoseconds (not µs) — fixes old bug
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

import pytest

from aurelius.connectors.dcgm import DCGMAdapter, _health_penalty
from aurelius.connectors.metric_mapping import dcgm_registry
from aurelius.connectors.prometheus import (
    FakePrometheusClient,
    PrometheusTelemetryConnector,
)
from aurelius.state.models import GPUState

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "prometheus")


def _snapshot_from_prom_file(filename: str):
    """Load a fixture file and return a TelemetrySnapshot via FakePrometheusClient."""
    with open(os.path.join(FIXTURES_DIR, filename)) as f:
        text = f.read()
    client = FakePrometheusClient(prometheus_text=text)
    reg = dcgm_registry()
    connector = PrometheusTelemetryConnector(client, reg, source="dcgm-test")
    return connector.fetch_snapshot()


def _snapshot_from_fixtures(fixtures: dict):
    """Build a TelemetrySnapshot from a raw fixtures dict via FakePrometheusClient."""
    client = FakePrometheusClient(fixtures=fixtures)
    reg = dcgm_registry()
    connector = PrometheusTelemetryConnector(client, reg, source="dcgm-test")
    return connector.fetch_snapshot()


class TestDCGMAdapterBasic:
    def test_discovers_gpu_uuids(self):
        snapshot = _snapshot_from_prom_file("dcgm_metrics.prom")
        adapter = DCGMAdapter()
        uuids = adapter.all_gpu_uuids(snapshot)
        assert "GPU-aaaa-bbbb-cccc-0000" in uuids
        assert "GPU-aaaa-bbbb-cccc-0001" in uuids
        assert "GPU-dddd-eeee-ffff-0000" in uuids
        assert len(uuids) == 3

    def test_normalize_single_gpu(self):
        snapshot = _snapshot_from_prom_file("dcgm_metrics.prom")
        adapter = DCGMAdapter()
        gpu = adapter.normalize_gpu_state(
            snapshot=snapshot,
            gpu_uuid="GPU-aaaa-bbbb-cccc-0000",
            node_id="node-01",
            region="us-east-1",
        )
        assert isinstance(gpu, GPUState)
        assert gpu.gpu_uuid == "GPU-aaaa-bbbb-cccc-0000"
        assert gpu.node_id == "node-01"
        assert gpu.region == "us-east-1"

    def test_util_pct_correct(self):
        snapshot = _snapshot_from_prom_file("dcgm_metrics.prom")
        adapter = DCGMAdapter()
        gpu0 = adapter.normalize_gpu_state(snapshot, "GPU-aaaa-bbbb-cccc-0000", "node-01", "us-east-1")
        gpu1 = adapter.normalize_gpu_state(snapshot, "GPU-aaaa-bbbb-cccc-0001", "node-01", "us-east-1")
        assert gpu0.util_pct == pytest.approx(72.5)
        assert gpu1.util_pct == pytest.approx(88.0)

    def test_power_correct(self):
        snapshot = _snapshot_from_prom_file("dcgm_metrics.prom")
        adapter = DCGMAdapter()
        gpu0 = adapter.normalize_gpu_state(snapshot, "GPU-aaaa-bbbb-cccc-0000", "node-01", "us-east-1")
        assert gpu0.power_w == pytest.approx(285.3)

    def test_temp_correct(self):
        snapshot = _snapshot_from_prom_file("dcgm_metrics.prom")
        adapter = DCGMAdapter()
        gpu0 = adapter.normalize_gpu_state(snapshot, "GPU-aaaa-bbbb-cccc-0000", "node-01", "us-east-1")
        assert gpu0.temp_c == pytest.approx(63.0)

    def test_mem_total_derived(self):
        snapshot = _snapshot_from_prom_file("dcgm_metrics.prom")
        adapter = DCGMAdapter()
        gpu0 = adapter.normalize_gpu_state(snapshot, "GPU-aaaa-bbbb-cccc-0000", "node-01", "us-east-1")
        # mem_total = used (61440) + free (20480) = 81920 (no reserved in fixture)
        assert gpu0.mem_total_mb == pytest.approx(61440 + 20480)

    def test_normalize_all_gpus(self):
        snapshot = _snapshot_from_prom_file("dcgm_metrics.prom")
        adapter = DCGMAdapter()
        gpus = adapter.normalize_gpus(snapshot, "node-01", "us-east-1")
        assert len(gpus) == 3
        for g in gpus:
            assert isinstance(g, GPUState)
            assert g.util_pct is not None  # fixture has util for all

    def test_timestamp_utc_aware(self):
        snapshot = _snapshot_from_prom_file("dcgm_metrics.prom")
        adapter = DCGMAdapter()
        gpu = adapter.normalize_gpu_state(snapshot, "GPU-aaaa-bbbb-cccc-0000", "n1", "us-east-1")
        assert gpu.timestamp.tzinfo is not None

    def test_provenance_source(self):
        snapshot = _snapshot_from_prom_file("dcgm_metrics.prom")
        adapter = DCGMAdapter()
        gpu = adapter.normalize_gpu_state(snapshot, "GPU-aaaa-bbbb-cccc-0000", "n1", "us-east-1")
        assert gpu.provenance.source == "dcgm-test"
        assert gpu.provenance.is_sandbox is True


class TestDCGMAdapterMissingMetrics:
    def test_missing_optional_fields_are_none_not_zero(self):
        """Critical: missing DCGM metrics must be None, never fabricated as 0."""
        snapshot = _snapshot_from_fixtures({
            "DCGM_FI_DEV_GPU_UTIL": [{"labels": {"UUID": "GPU-abc", "gpu": "0", "node": "n1"}, "value": 80.0}],
        })
        adapter = DCGMAdapter()
        gpu = adapter.normalize_gpu_state(snapshot, "GPU-abc", "n1", "us-east-1")
        assert gpu.util_pct == pytest.approx(80.0)

        # These are missing — must be None, not 0
        assert gpu.power_w is None
        assert gpu.temp_c is None
        assert gpu.mem_used_mb is None
        assert gpu.mem_total_mb is None
        assert gpu.ecc_sbe_total is None
        assert gpu.xid_last is None
        assert gpu.nvlink_tx_bytes_per_s is None
        assert gpu.clocks_event_reasons is None

    def test_empty_snapshot_returns_no_gpus(self):
        snapshot = _snapshot_from_fixtures({})
        adapter = DCGMAdapter()
        gpus = adapter.normalize_gpus(snapshot, "n1", "us-east-1")
        assert gpus == []

    def test_unknown_metrics_list(self):
        snapshot = _snapshot_from_fixtures({
            "DCGM_FI_DEV_GPU_UTIL": [{"labels": {"UUID": "GPU-abc"}, "value": 50.0}],
        })
        adapter = DCGMAdapter()
        unknown = adapter.unknown_metrics(snapshot)
        assert isinstance(unknown, list)
        # Many DCGM metrics are not in the fixtures
        assert len(unknown) > 0


class TestDCGMAdapterHealthPenalty:
    def _make_gpu(self, util=None, temp=None, thermal_ns=None, ecc_dbe=None) -> GPUState:
        ts = datetime.now(tz=timezone.utc)
        from aurelius.state.models import Provenance
        prov = Provenance(source="test", fetched_at=ts, confidence="high", is_sandbox=True)
        return GPUState(
            gpu_uuid="test-uuid",
            node_id="n1",
            region="r1",
            timestamp=ts,
            provenance=prov,
            util_pct=util,
            temp_c=temp,
            thermal_violation_ns=thermal_ns,
            ecc_dbe_total=ecc_dbe,
        )

    def test_healthy_gpu_low_penalty(self):
        gpu = self._make_gpu(util=50.0, temp=65.0)
        hp = _health_penalty(gpu)
        assert hp is not None
        assert hp < 0.2

    def test_high_util_increases_penalty(self):
        gpu_normal = self._make_gpu(util=50.0, temp=60.0)
        gpu_high = self._make_gpu(util=95.0, temp=60.0)
        hp_normal = _health_penalty(gpu_normal)
        hp_high = _health_penalty(gpu_high)
        assert hp_high > hp_normal

    def test_high_temp_increases_penalty(self):
        gpu_cool = self._make_gpu(util=50.0, temp=50.0)
        gpu_hot = self._make_gpu(util=50.0, temp=88.0)
        hp_cool = _health_penalty(gpu_cool)
        hp_hot = _health_penalty(gpu_hot)
        assert hp_hot > hp_cool

    def test_critical_temp_high_penalty(self):
        gpu = self._make_gpu(util=50.0, temp=96.0)
        hp = _health_penalty(gpu)
        assert hp is not None
        assert hp >= 0.5

    def test_thermal_violation_ns_increases_penalty(self):
        """Thermal violation is in nanoseconds — 1e9 ns = 1 second."""
        gpu_no_throttle = self._make_gpu(util=70.0, temp=70.0, thermal_ns=0)
        gpu_throttled = self._make_gpu(util=70.0, temp=70.0, thermal_ns=500_000_000)  # 0.5 seconds
        hp_no = _health_penalty(gpu_no_throttle)
        hp_throttled = _health_penalty(gpu_throttled)
        assert hp_throttled > hp_no

    def test_ecc_dbe_increases_penalty(self):
        gpu_ok = self._make_gpu(util=50.0, temp=60.0, ecc_dbe=0)
        gpu_ecc = self._make_gpu(util=50.0, temp=60.0, ecc_dbe=3)
        hp_ok = _health_penalty(gpu_ok)
        hp_ecc = _health_penalty(gpu_ecc)
        assert hp_ecc > hp_ok

    def test_no_data_returns_none(self):
        gpu = self._make_gpu(util=None, temp=None)
        assert _health_penalty(gpu) is None

    def test_penalty_capped_at_one(self):
        gpu = self._make_gpu(util=100.0, temp=98.0, thermal_ns=2_000_000_000, ecc_dbe=10)
        hp = _health_penalty(gpu)
        assert hp is not None
        assert hp <= 1.0


class TestDCGMAdapterThermalThrottling:
    def _gpu_with_clocks_event(self, bitmask: int) -> GPUState:
        ts = datetime.now(tz=timezone.utc)
        from aurelius.state.models import Provenance
        prov = Provenance(source="test", fetched_at=ts, confidence="high", is_sandbox=True)
        return GPUState(
            gpu_uuid="test",
            node_id="n1",
            region="r1",
            timestamp=ts,
            provenance=prov,
            clocks_event_reasons=bitmask,
        )

    def test_no_throttle(self):
        gpu = self._gpu_with_clocks_event(0)
        assert gpu.thermal_throttling is False

    def test_sw_thermal_throttle(self):
        gpu = self._gpu_with_clocks_event(0x20)  # SW_THERMAL
        assert gpu.thermal_throttling is True

    def test_hw_thermal_throttle(self):
        gpu = self._gpu_with_clocks_event(0x40)  # HW_THERMAL
        assert gpu.thermal_throttling is True

    def test_sw_power_cap_throttle(self):
        gpu = self._gpu_with_clocks_event(0x04)  # SW_POWER_CAP
        assert gpu.thermal_throttling is True

    def test_clocks_event_none(self):
        ts = datetime.now(tz=timezone.utc)
        from aurelius.state.models import Provenance
        prov = Provenance(source="test", fetched_at=ts, confidence="high", is_sandbox=True)
        gpu = GPUState(
            gpu_uuid="test",
            node_id="n1",
            region="r1",
            timestamp=ts,
            provenance=prov,
            clocks_event_reasons=None,
        )
        assert gpu.thermal_throttling is None


class TestDCGMAdapterPCIETraffic:
    def test_pcie_bytes_parsed_correctly(self):
        snapshot = _snapshot_from_prom_file("dcgm_metrics.prom")
        adapter = DCGMAdapter()
        gpu0 = adapter.normalize_gpu_state(snapshot, "GPU-aaaa-bbbb-cccc-0000", "n1", "us-east-1")
        assert gpu0.pcie_tx_bytes_per_s == pytest.approx(1073741824)
        assert gpu0.pcie_rx_bytes_per_s == pytest.approx(2147483648)

    def test_missing_pcie_is_none(self):
        snapshot = _snapshot_from_fixtures({
            "DCGM_FI_DEV_GPU_UTIL": [{"labels": {"UUID": "GPU-x"}, "value": 50.0}],
        })
        adapter = DCGMAdapter()
        gpu = adapter.normalize_gpu_state(snapshot, "GPU-x", "n1", "us-east-1")
        assert gpu.pcie_tx_bytes_per_s is None
        assert gpu.pcie_rx_bytes_per_s is None


class TestDCGMAdapterGPUIndex:
    def test_gpu_index_extracted_from_labels(self):
        snapshot = _snapshot_from_prom_file("dcgm_metrics.prom")
        adapter = DCGMAdapter()
        gpu0 = adapter.normalize_gpu_state(snapshot, "GPU-aaaa-bbbb-cccc-0000", "n1", "us-east-1")
        assert gpu0.gpu_index == 0

    def test_gpu_type_extracted_from_labels(self):
        snapshot = _snapshot_from_prom_file("dcgm_metrics.prom")
        adapter = DCGMAdapter()
        gpu0 = adapter.normalize_gpu_state(snapshot, "GPU-aaaa-bbbb-cccc-0000", "n1", "us-east-1")
        assert gpu0.gpu_type == "A100-SXM4-80GB"
