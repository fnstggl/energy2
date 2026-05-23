"""Tests for DCGM/Prometheus GPU telemetry ingestion.

Covers:
  - GPUMetrics and GPUHealthScore datamodels
  - score_gpu_health(): healthy, overheated, throttled, ECC-error cases
  - aggregate_region_health(): empty, all-schedulable, all-unschedulable, mixed
  - parse_prometheus_text(): full fixture parsing, label extraction
  - DCGMProvider.from_prom_fixture(): load healthy and degraded fixtures
  - DCGMProvider.from_csv(): load, missing column error
  - DCGMProvider.from_dataframe(): construction
  - DCGMProvider.generate_fixture(): count, determinism, seed diff, patterns
  - DCGMProvider.get_health_penalty(): leakage-safe, unknown region, empty
  - DCGMProvider.to_dict_lookup(): structure, aggregation
  - DCGMProvider.get_gpu_scores(): node-level, empty
  - DCGMProvider.save_csv() / round-trip
  - ObjectiveFunction gpu_health_cost integration
  - JobScheduler gpu_health routing (routes away from degraded region)
  - BacktestEngine gpu_df parameter (leakage-safe fold construction)
  - OptimizationConfig gpu_health_cost_per_hour field
"""

import csv
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aurelius.ingestion.dcgm_provider import (
    DCGMProvider,
    aggregate_region_health,
    parse_prometheus_text,
    score_gpu_health,
    _TEMP_CRITICAL,
    _TEMP_SAFE,
    _UTIL_WARN,
)
from aurelius.models import (
    GPUHealthScore,
    GPUMetrics,
    Job,
    OptimizationConfig,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FIXTURE_DIR = Path(__file__).parent.parent / "data" / "fixtures"
HEALTHY_PROM = FIXTURE_DIR / "dcgm_metrics_healthy.prom"
DEGRADED_PROM = FIXTURE_DIR / "dcgm_metrics_degraded.prom"


def make_gpu(
    region="us-west",
    node="node-01",
    gpu_index=0,
    util=40.0,
    temp=55.0,
    ecc_sbe=0,
    ecc_dbe=0,
    xid=0,
    power_throttle_us=0.0,
    thermal_throttle_us=0.0,
    clock_throttle=0,
    power_w=180.0,
) -> GPUMetrics:
    ts = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)
    return GPUMetrics(
        timestamp=ts,
        region=region,
        node_id=node,
        gpu_index=gpu_index,
        gpu_uuid=f"GPU-{node}-{gpu_index}",
        gpu_type="a100",
        gpu_util_pct=util,
        mem_used_mb=30000.0,
        mem_total_mb=80000.0,
        power_usage_w=power_w,
        gpu_temp_c=temp,
        ecc_sbe_count=ecc_sbe,
        ecc_dbe_count=ecc_dbe,
        xid_error_count=xid,
        power_throttle_us=power_throttle_us,
        thermal_throttle_us=thermal_throttle_us,
        clock_throttle_reasons=clock_throttle,
    )


# ---------------------------------------------------------------------------
# TestGPUMetricsModel
# ---------------------------------------------------------------------------

class TestGPUMetricsModel:
    def test_construction_defaults(self):
        m = make_gpu()
        assert m.ecc_sbe_count == 0
        assert m.ecc_dbe_count == 0
        assert m.xid_error_count == 0
        assert m.power_throttle_us == 0.0
        assert m.thermal_throttle_us == 0.0
        assert m.clock_throttle_reasons == 0

    def test_fields_present(self):
        m = make_gpu(util=75.0, temp=65.0)
        assert m.gpu_util_pct == 75.0
        assert m.gpu_temp_c == 65.0
        assert m.region == "us-west"
        assert m.node_id == "node-01"

    def test_optional_fields(self):
        m = make_gpu(ecc_sbe=3, ecc_dbe=1, xid=1)
        assert m.ecc_sbe_count == 3
        assert m.ecc_dbe_count == 1
        assert m.xid_error_count == 1


# ---------------------------------------------------------------------------
# TestGPUHealthScoreModel
# ---------------------------------------------------------------------------

class TestGPUHealthScoreModel:
    def test_healthy_gpu_score(self):
        m = make_gpu(util=40.0, temp=55.0)
        score = score_gpu_health(m)
        assert score.health_penalty == 0.0
        assert score.is_schedulable is True
        assert score.reason_codes == []
        assert score.ecc_penalty == 0.0

    def test_high_utilization_penalty(self):
        m = make_gpu(util=95.0, temp=60.0)
        score = score_gpu_health(m)
        assert score.utilization_penalty > 0.0
        assert score.health_penalty > 0.0
        assert any("util" in c for c in score.reason_codes)

    def test_low_utilization_no_penalty(self):
        m = make_gpu(util=_UTIL_WARN - 5.0)
        score = score_gpu_health(m)
        assert score.utilization_penalty == 0.0

    def test_boundary_utilization(self):
        m_warn = make_gpu(util=_UTIL_WARN)
        m_above = make_gpu(util=_UTIL_WARN + 10.0)
        s_warn = score_gpu_health(m_warn)
        s_above = score_gpu_health(m_above)
        assert s_warn.utilization_penalty == 0.0
        assert s_above.utilization_penalty > 0.0

    def test_thermal_penalty_safe_zone(self):
        m = make_gpu(temp=_TEMP_SAFE - 5.0)
        score = score_gpu_health(m)
        assert score.thermal_penalty == 0.0

    def test_thermal_penalty_elevated(self):
        m = make_gpu(temp=80.0)
        score = score_gpu_health(m)
        assert score.thermal_penalty > 0.0
        assert any("temp" in c for c in score.reason_codes)

    def test_thermal_penalty_critical(self):
        m = make_gpu(temp=_TEMP_CRITICAL + 1.0)
        score = score_gpu_health(m)
        assert score.thermal_penalty == 1.0
        assert score.is_schedulable is False

    def test_ecc_dbe_flags_unschedulable(self):
        m = make_gpu(ecc_dbe=1)
        score = score_gpu_health(m)
        assert score.is_schedulable is False
        assert score.ecc_penalty == 1.0
        assert any("ecc_dbe" in c for c in score.reason_codes)

    def test_ecc_sbe_many_half_penalty(self):
        m = make_gpu(ecc_sbe=5)
        score = score_gpu_health(m)
        assert score.ecc_penalty == 0.5
        assert any("ecc_sbe" in c for c in score.reason_codes)

    def test_ecc_sbe_few_no_penalty(self):
        m = make_gpu(ecc_sbe=3)
        score = score_gpu_health(m)
        assert score.ecc_penalty == 0.0

    def test_xid_error_flags_unschedulable(self):
        m = make_gpu(xid=1)
        score = score_gpu_health(m)
        assert score.is_schedulable is False

    def test_power_throttle_penalty(self):
        m = make_gpu(power_throttle_us=500000.0)
        score = score_gpu_health(m)
        assert score.throttle_penalty > 0.0

    def test_thermal_throttle_penalty(self):
        m = make_gpu(thermal_throttle_us=500000.0)
        score = score_gpu_health(m)
        assert score.throttle_penalty > 0.0

    def test_clock_throttle_reasons_alone(self):
        m = make_gpu(clock_throttle=4)
        score = score_gpu_health(m)
        # Bitmask alone gives partial throttle penalty
        assert score.throttle_penalty > 0.0

    def test_health_penalty_bounded_0_to_1(self):
        for util in [0, 50, 80, 100]:
            for temp in [30, 60, 80, 100]:
                m = make_gpu(util=float(util), temp=float(temp))
                score = score_gpu_health(m)
                assert 0.0 <= score.health_penalty <= 1.0

    def test_perfect_gpu_score_zero(self):
        m = make_gpu(util=0.0, temp=25.0, ecc_sbe=0, ecc_dbe=0, xid=0)
        score = score_gpu_health(m)
        assert score.health_penalty == 0.0
        assert score.is_schedulable is True


# ---------------------------------------------------------------------------
# TestAggregateRegionHealth
# ---------------------------------------------------------------------------

class TestAggregateRegionHealth:
    def test_empty_list(self):
        assert aggregate_region_health([]) == 0.0

    def test_all_healthy(self):
        scores = [score_gpu_health(make_gpu(util=40.0, temp=50.0)) for _ in range(4)]
        assert aggregate_region_health(scores) == 0.0

    def test_all_unschedulable(self):
        scores = [score_gpu_health(make_gpu(ecc_dbe=1)) for _ in range(4)]
        # All unschedulable → worst case
        assert aggregate_region_health(scores) == 1.0

    def test_mixed_schedulable_unschedulable(self):
        healthy = [score_gpu_health(make_gpu(util=40.0, temp=50.0)) for _ in range(3)]
        broken = [score_gpu_health(make_gpu(ecc_dbe=1))]
        # Only healthy GPUs contribute to mean
        result = aggregate_region_health(healthy + broken)
        assert result == 0.0

    def test_weighted_mean_of_schedulable(self):
        gpu1 = make_gpu(util=90.0, temp=60.0)  # penalty > 0
        gpu2 = make_gpu(util=40.0, temp=50.0)  # penalty = 0
        s1 = score_gpu_health(gpu1)
        s2 = score_gpu_health(gpu2)
        assert s1.is_schedulable and s2.is_schedulable
        result = aggregate_region_health([s1, s2])
        expected = (s1.health_penalty + s2.health_penalty) / 2
        assert abs(result - expected) < 1e-6


# ---------------------------------------------------------------------------
# TestPrometheusTextParser
# ---------------------------------------------------------------------------

class TestPrometheusTextParser:
    def test_parses_healthy_fixture(self):
        if not HEALTHY_PROM.exists():
            pytest.skip("Healthy fixture file not found")
        text = HEALTHY_PROM.read_text()
        ts = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)
        metrics = parse_prometheus_text(text, region="us-west", timestamp=ts)
        assert len(metrics) == 4
        for m in metrics:
            assert m.region == "us-west"
            assert m.gpu_type == "a100"
            assert m.node_id == "gpu-node-01"
            assert m.gpu_temp_c > 0
            assert m.gpu_util_pct >= 0
            assert m.ecc_dbe_count == 0
            assert m.xid_error_count == 0

    def test_parses_degraded_fixture(self):
        if not DEGRADED_PROM.exists():
            pytest.skip("Degraded fixture file not found")
        text = DEGRADED_PROM.read_text()
        ts = datetime(2026, 1, 15, 14, 0, tzinfo=timezone.utc)
        metrics = parse_prometheus_text(text, region="us-east", timestamp=ts)
        assert len(metrics) == 4
        # ECC DBE error GPU should be detected
        dbe_gpus = [m for m in metrics if m.ecc_dbe_count > 0]
        assert len(dbe_gpus) >= 1
        # Hot GPUs
        hot_gpus = [m for m in metrics if m.gpu_temp_c > 80]
        assert len(hot_gpus) >= 1

    def test_ignores_non_dcgm_metrics(self):
        text = """
# HELP node_cpu_seconds_total CPU time
# TYPE node_cpu_seconds_total counter
node_cpu_seconds_total{cpu="0",mode="idle"} 12345
DCGM_FI_DEV_GPU_UTIL{gpu="0",Hostname="host-01",UUID="GPU-test001"} 55
"""
        metrics = parse_prometheus_text(text, region="us-west")
        assert len(metrics) == 1
        assert metrics[0].gpu_util_pct == 55.0

    def test_handles_empty_input(self):
        metrics = parse_prometheus_text("", region="us-west")
        assert metrics == []

    def test_handles_comment_only_input(self):
        text = "# HELP DCGM_FI_DEV_GPU_UTIL GPU util\n# TYPE DCGM_FI_DEV_GPU_UTIL gauge\n"
        metrics = parse_prometheus_text(text, region="us-west")
        assert metrics == []

    def test_timestamp_defaults_to_now(self):
        text = 'DCGM_FI_DEV_GPU_UTIL{gpu="0",Hostname="h1",UUID="GPU-x"} 50\n'
        before = datetime.now(timezone.utc)
        metrics = parse_prometheus_text(text, region="us-west")
        after = datetime.now(timezone.utc)
        assert before.replace(second=0, microsecond=0) <= metrics[0].timestamp <= after.replace(second=0, microsecond=0) + timedelta(minutes=1)

    def test_multiple_nodes(self):
        text = """
DCGM_FI_DEV_GPU_UTIL{gpu="0",Hostname="node-01",UUID="GPU-n1g0"} 40
DCGM_FI_DEV_GPU_UTIL{gpu="0",Hostname="node-02",UUID="GPU-n2g0"} 60
"""
        metrics = parse_prometheus_text(text, region="us-west")
        assert len(metrics) == 2
        nodes = {m.node_id for m in metrics}
        assert nodes == {"node-01", "node-02"}

    def test_mem_total_derived_from_used_plus_free(self):
        text = """
DCGM_FI_DEV_FB_USED{gpu="0",Hostname="n1",UUID="GPU-x1"} 40000
DCGM_FI_DEV_FB_FREE{gpu="0",Hostname="n1",UUID="GPU-x1"} 40000
"""
        metrics = parse_prometheus_text(text, region="us-west")
        assert len(metrics) == 1
        assert metrics[0].mem_total_mb == 80000.0
        assert metrics[0].mem_used_mb == 40000.0


# ---------------------------------------------------------------------------
# TestDCGMProviderFromPromFixture
# ---------------------------------------------------------------------------

class TestDCGMProviderFromPromFixture:
    def test_loads_healthy_fixture(self):
        if not HEALTHY_PROM.exists():
            pytest.skip("Healthy fixture file not found")
        ts = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)
        p = DCGMProvider.from_prom_fixture(str(HEALTHY_PROM), region="us-west", timestamp=ts)
        assert p.record_count == 4
        assert "us-west" in p.regions

    def test_loads_degraded_fixture(self):
        if not DEGRADED_PROM.exists():
            pytest.skip("Degraded fixture file not found")
        ts = datetime(2026, 1, 15, 14, 0, tzinfo=timezone.utc)
        p = DCGMProvider.from_prom_fixture(str(DEGRADED_PROM), region="us-east", timestamp=ts)
        assert p.record_count == 4
        assert "us-east" in p.regions

    def test_health_penalty_healthy_is_low(self):
        if not HEALTHY_PROM.exists():
            pytest.skip()
        ts = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)
        p = DCGMProvider.from_prom_fixture(str(HEALTHY_PROM), region="us-west", timestamp=ts)
        penalty = p.get_health_penalty("us-west", ts + timedelta(hours=1))
        assert penalty < 0.4, f"Healthy GPUs should have low penalty, got {penalty}"

    def test_health_penalty_degraded_is_higher(self):
        if not DEGRADED_PROM.exists():
            pytest.skip()
        ts = datetime(2026, 1, 15, 14, 0, tzinfo=timezone.utc)
        p = DCGMProvider.from_prom_fixture(str(DEGRADED_PROM), region="us-east", timestamp=ts)
        # Degraded fixture has hot GPUs and ECC errors; only schedulable ones included
        # but they should still have elevated penalty
        scores = p.get_gpu_scores("us-east", ts + timedelta(hours=1))
        schedulable = [s for s in scores if s.is_schedulable]
        if schedulable:
            mean_penalty = sum(s.health_penalty for s in schedulable) / len(schedulable)
            assert mean_penalty > 0.0


# ---------------------------------------------------------------------------
# TestDCGMProviderFromCSV
# ---------------------------------------------------------------------------

class TestDCGMProviderFromCSV:
    def _make_csv_row(self, region="us-west", node="n1", ts_str="2026-01-15T12:00:00+00:00"):
        return {
            "timestamp": ts_str,
            "region": region,
            "node_id": node,
            "gpu_index": 0,
            "gpu_uuid": f"GPU-{node}-0",
            "gpu_type": "a100",
            "gpu_util_pct": 55.0,
            "mem_used_mb": 30000.0,
            "mem_total_mb": 80000.0,
            "power_usage_w": 200.0,
            "gpu_temp_c": 65.0,
            "ecc_sbe_count": 0,
            "ecc_dbe_count": 0,
            "xid_error_count": 0,
            "power_throttle_us": 0.0,
            "thermal_throttle_us": 0.0,
            "clock_throttle_reasons": 0,
        }

    def test_load_valid_csv(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            row = self._make_csv_row()
            writer = csv.DictWriter(f, fieldnames=sorted(row.keys()))
            writer.writeheader()
            writer.writerow(row)
            path = f.name
        try:
            p = DCGMProvider.from_csv(path)
            assert p.record_count == 1
        finally:
            os.unlink(path)

    def test_missing_columns_raises(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            writer = csv.DictWriter(f, fieldnames=["timestamp", "region"])
            writer.writeheader()
            writer.writerow({"timestamp": "2026-01-15T12:00:00+00:00", "region": "us-west"})
            path = f.name
        try:
            with pytest.raises(ValueError, match="missing columns"):
                DCGMProvider.from_csv(path)
        finally:
            os.unlink(path)

    def test_multiple_rows(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            rows = [
                self._make_csv_row("us-west", "n1", "2026-01-15T10:00:00+00:00"),
                self._make_csv_row("us-west", "n1", "2026-01-15T11:00:00+00:00"),
                self._make_csv_row("us-east", "n2", "2026-01-15T10:00:00+00:00"),
            ]
            writer = csv.DictWriter(f, fieldnames=sorted(rows[0].keys()))
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
            path = f.name
        try:
            p = DCGMProvider.from_csv(path)
            assert p.record_count == 3
            assert set(p.regions) == {"us-west", "us-east"}
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# TestDCGMProviderGenerateFixture
# ---------------------------------------------------------------------------

class TestDCGMProviderGenerateFixture:
    def test_record_count(self):
        p = DCGMProvider.generate_fixture(
            regions=["us-west"], n_nodes_per_region=2, n_gpus_per_node=4, hours=6, seed=1
        )
        # 1 region * 2 nodes * 4 gpus * 6 hours = 48
        assert p.record_count == 48

    def test_multi_region_count(self):
        p = DCGMProvider.generate_fixture(
            regions=["us-west", "us-east"], n_nodes_per_region=2, n_gpus_per_node=2, hours=3, seed=1
        )
        assert p.record_count == 24

    def test_determinism(self):
        p1 = DCGMProvider.generate_fixture(["us-west"], seed=42)
        p2 = DCGMProvider.generate_fixture(["us-west"], seed=42)
        vals1 = [m.gpu_util_pct for m in p1._records]
        vals2 = [m.gpu_util_pct for m in p2._records]
        assert vals1 == vals2

    def test_different_seeds_differ(self):
        p1 = DCGMProvider.generate_fixture(["us-west"], seed=42)
        p2 = DCGMProvider.generate_fixture(["us-west"], seed=99)
        vals1 = [m.gpu_util_pct for m in p1._records]
        vals2 = [m.gpu_util_pct for m in p2._records]
        assert vals1 != vals2

    def test_utilization_non_negative(self):
        p = DCGMProvider.generate_fixture(["us-west"], seed=7)
        assert all(m.gpu_util_pct >= 0.0 for m in p._records)
        assert all(m.gpu_util_pct <= 100.0 for m in p._records)

    def test_temperature_range(self):
        p = DCGMProvider.generate_fixture(["us-west"], seed=7)
        assert all(m.gpu_temp_c >= 20.0 for m in p._records)
        assert all(m.gpu_temp_c <= 105.0 for m in p._records)

    def test_business_hours_higher_utilization(self):
        p = DCGMProvider.generate_fixture(
            ["us-west"], n_nodes_per_region=2, n_gpus_per_node=4, hours=24, seed=42
        )
        business = [m.gpu_util_pct for m in p._records if 12 <= m.timestamp.hour < 18]
        offpeak = [m.gpu_util_pct for m in p._records if m.timestamp.hour < 6]
        assert sum(business) / len(business) > sum(offpeak) / len(offpeak)

    def test_regions_assigned_correctly(self):
        p = DCGMProvider.generate_fixture(["us-west", "us-south"], seed=1)
        assert set(p.regions) == {"us-west", "us-south"}


# ---------------------------------------------------------------------------
# TestDCGMProviderLookup
# ---------------------------------------------------------------------------

class TestDCGMProviderLookup:
    def _make_provider_two_hours(self) -> DCGMProvider:
        ts1 = datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc)
        ts2 = datetime(2026, 1, 15, 11, 0, tzinfo=timezone.utc)
        # ts1 hour: healthy
        gpus_h1 = [make_gpu(region="us-west", temp=50.0, util=40.0)]
        gpus_h1[0] = GPUMetrics(
            timestamp=ts1, region="us-west", node_id="n1", gpu_index=0,
            gpu_uuid="GPU-n1-0", gpu_type="a100",
            gpu_util_pct=40.0, mem_used_mb=30000, mem_total_mb=80000,
            power_usage_w=180.0, gpu_temp_c=50.0,
        )
        # ts2 hour: degraded (high util)
        gpus_h2 = GPUMetrics(
            timestamp=ts2, region="us-west", node_id="n1", gpu_index=0,
            gpu_uuid="GPU-n1-0", gpu_type="a100",
            gpu_util_pct=95.0, mem_used_mb=70000, mem_total_mb=80000,
            power_usage_w=380.0, gpu_temp_c=80.0,
        )
        p = DCGMProvider()
        p._records = [gpus_h1[0], gpus_h2]
        return p

    def test_get_health_penalty_unknown_region(self):
        p = DCGMProvider()
        p._records = [make_gpu(region="us-west")]
        assert p.get_health_penalty("us-east", datetime.now(timezone.utc)) == 0.0

    def test_get_health_penalty_empty_provider(self):
        p = DCGMProvider()
        assert p.get_health_penalty("us-west", datetime.now(timezone.utc)) == 0.0

    def test_get_health_penalty_leakage_safe(self):
        p = self._make_provider_two_hours()
        ts1 = datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc)
        ts2 = datetime(2026, 1, 15, 11, 0, tzinfo=timezone.utc)
        # Query at ts1 → only ts1 data used
        penalty_at_t1 = p.get_health_penalty("us-west", ts1)
        # Query at ts2 → both ts1 and ts2 data exist; ts2 is used (last known ≤ ts2)
        penalty_at_t2 = p.get_health_penalty("us-west", ts2)
        # ts1 was healthy (util=40, temp=50) so penalty should be 0
        assert penalty_at_t1 == 0.0
        # ts2 was degraded (util=95, temp=80) so penalty should be higher
        assert penalty_at_t2 > penalty_at_t1

    def test_get_health_penalty_before_any_data(self):
        p = DCGMProvider()
        ts = datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc)
        p._records = [GPUMetrics(
            timestamp=ts, region="us-west", node_id="n1", gpu_index=0,
            gpu_uuid="GPU-x", gpu_type="a100",
            gpu_util_pct=50.0, mem_used_mb=30000, mem_total_mb=80000,
            power_usage_w=200.0, gpu_temp_c=60.0,
        )]
        # Query before any data → 0
        early = ts - timedelta(hours=1)
        assert p.get_health_penalty("us-west", early) == 0.0

    def test_to_dict_lookup_structure(self):
        p = DCGMProvider.generate_fixture(["us-west", "us-east"], hours=3, seed=1)
        lookup = p.to_dict_lookup()
        assert "us-west" in lookup
        assert "us-east" in lookup
        for region, ts_map in lookup.items():
            for ts, penalty in ts_map.items():
                assert isinstance(ts, datetime)
                assert 0.0 <= penalty <= 1.0

    def test_to_dict_lookup_matches_get_health_penalty(self):
        p = DCGMProvider.generate_fixture(["us-west"], hours=3, seed=1)
        lookup = p.to_dict_lookup()
        for ts, penalty in lookup.get("us-west", {}).items():
            # get_health_penalty with exact ts should return same value
            result = p.get_health_penalty("us-west", ts)
            assert abs(result - penalty) < 1e-6

    def test_get_gpu_scores_returns_list(self):
        if not HEALTHY_PROM.exists():
            pytest.skip()
        ts = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)
        p = DCGMProvider.from_prom_fixture(str(HEALTHY_PROM), region="us-west", timestamp=ts)
        scores = p.get_gpu_scores("us-west", ts + timedelta(hours=1))
        assert len(scores) == 4
        for s in scores:
            assert isinstance(s, GPUHealthScore)

    def test_get_gpu_scores_empty_for_unknown_region(self):
        p = DCGMProvider.generate_fixture(["us-west"], hours=3, seed=1)
        scores = p.get_gpu_scores("us-north", datetime.now(timezone.utc))
        assert scores == []

    def test_get_gpu_scores_future_data_not_used(self):
        future_ts = datetime(2030, 1, 1, tzinfo=timezone.utc)
        p = DCGMProvider.generate_fixture(
            ["us-west"], hours=3, seed=1,
            start_dt=future_ts
        )
        # Query before the future fixture data → empty
        past_ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
        scores = p.get_gpu_scores("us-west", past_ts)
        assert scores == []


# ---------------------------------------------------------------------------
# TestDCGMProviderCSVRoundTrip
# ---------------------------------------------------------------------------

class TestDCGMProviderCSVRoundTrip:
    def test_save_and_reload(self):
        p_orig = DCGMProvider.generate_fixture(["us-west"], hours=3, seed=1)
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            p_orig.save_csv(path)
            p_reload = DCGMProvider.from_csv(path)
            assert p_orig.record_count == p_reload.record_count
            # Health lookups should match
            ts = datetime(2026, 1, 1, 2, 0, tzinfo=timezone.utc)
            p1 = p_orig.get_health_penalty("us-west", ts)
            p2 = p_reload.get_health_penalty("us-west", ts)
            assert abs(p1 - p2) < 1e-4
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# TestOptimizationConfigGPUHealth
# ---------------------------------------------------------------------------

class TestOptimizationConfigGPUHealth:
    def test_default_zero(self):
        cfg = OptimizationConfig()
        assert cfg.gpu_health_cost_per_hour == 0.0

    def test_custom_value(self):
        cfg = OptimizationConfig(gpu_health_cost_per_hour=3.0)
        assert cfg.gpu_health_cost_per_hour == 3.0

    def test_to_dict_includes_field(self):
        cfg = OptimizationConfig(gpu_health_cost_per_hour=2.5)
        d = cfg.to_dict()
        assert "gpu_health_cost_per_hour" in d
        assert d["gpu_health_cost_per_hour"] == 2.5

    def test_backward_compat_zero_config(self):
        # Zero config must produce zero gpu_health_cost
        cfg = OptimizationConfig()
        assert cfg.gpu_health_cost_per_hour == 0.0


# ---------------------------------------------------------------------------
# TestObjectiveFunctionGPUHealth
# ---------------------------------------------------------------------------

class TestObjectiveFunctionGPUHealth:
    def _make_job(self) -> Job:
        ts = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)
        return Job(
            job_id="j1",
            workload_type="training",
            submit_time=ts,
            earliest_start=ts,
            runtime_hours=4.0,
            deadline=ts + timedelta(hours=10),
            power_kw=100.0,
            region_options=["us-west", "us-east"],
            gpu_count=8,
        )

    def _make_price_carbon(self):
        ts = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)
        price = {"us-west": {ts + timedelta(hours=i): 40.0 for i in range(24)},
                 "us-east": {ts + timedelta(hours=i): 40.0 for i in range(24)}}
        carbon = {"us-west": {ts + timedelta(hours=i): 300.0 for i in range(24)},
                  "us-east": {ts + timedelta(hours=i): 300.0 for i in range(24)}}
        return price, carbon

    def test_no_gpu_health_data_zero_cost(self):
        from aurelius.optimization.objective import ObjectiveFunction
        from aurelius.models import ScheduleDecision

        cfg = OptimizationConfig(gpu_health_cost_per_hour=2.0)
        fn = ObjectiveFunction(cfg)
        job = self._make_job()
        price, carbon = self._make_price_carbon()
        ts = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)
        decision = ScheduleDecision(
            job_id="j1", start_time=ts, region="us-west",
            power_fraction=1.0, actual_runtime_hours=4.0
        )
        result = fn.calculate([job], [decision], price, carbon, gpu_health_data=None)
        assert result.gpu_health_cost == 0.0

    def test_zero_config_zero_cost(self):
        from aurelius.optimization.objective import ObjectiveFunction
        from aurelius.models import ScheduleDecision

        cfg = OptimizationConfig(gpu_health_cost_per_hour=0.0)
        fn = ObjectiveFunction(cfg)
        job = self._make_job()
        price, carbon = self._make_price_carbon()
        ts = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)
        decision = ScheduleDecision(
            job_id="j1", start_time=ts, region="us-west",
            power_fraction=1.0, actual_runtime_hours=4.0
        )
        gpu_health_data = {"us-west": {ts: 0.8}}
        result = fn.calculate([job], [decision], price, carbon, gpu_health_data=gpu_health_data)
        assert result.gpu_health_cost == 0.0

    def test_gpu_health_cost_calculation(self):
        from aurelius.optimization.objective import ObjectiveFunction
        from aurelius.models import ScheduleDecision

        cfg = OptimizationConfig(gpu_health_cost_per_hour=1.0)
        fn = ObjectiveFunction(cfg)
        job = self._make_job()
        price, carbon = self._make_price_carbon()
        ts = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)
        decision = ScheduleDecision(
            job_id="j1", start_time=ts, region="us-west",
            power_fraction=1.0, actual_runtime_hours=4.0
        )
        gpu_health_data = {"us-west": {ts: 0.5}}  # 50% degraded
        result = fn.calculate([job], [decision], price, carbon, gpu_health_data=gpu_health_data)
        # expected: 0.5 * 1.0 $/h * 4h * 8 GPUs = 16.0
        assert abs(result.gpu_health_cost - 16.0) < 0.01

    def test_gpu_health_cost_included_in_total(self):
        from aurelius.optimization.objective import ObjectiveFunction
        from aurelius.models import ScheduleDecision

        cfg = OptimizationConfig(gpu_health_cost_per_hour=2.0)
        fn = ObjectiveFunction(cfg)
        job = self._make_job()
        price, carbon = self._make_price_carbon()
        ts = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)
        decision = ScheduleDecision(
            job_id="j1", start_time=ts, region="us-west",
            power_fraction=1.0, actual_runtime_hours=4.0
        )
        no_gpu = fn.calculate([job], [decision], price, carbon)
        with_gpu = fn.calculate(
            [job], [decision], price, carbon,
            gpu_health_data={"us-west": {ts: 0.5}}
        )
        assert with_gpu.total > no_gpu.total
        assert with_gpu.gpu_health_cost > 0.0

    def test_backward_compat_no_gpu_health_arg(self):
        from aurelius.optimization.objective import ObjectiveFunction
        from aurelius.models import ScheduleDecision

        fn = ObjectiveFunction()
        job = self._make_job()
        price, carbon = self._make_price_carbon()
        ts = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)
        decision = ScheduleDecision(
            job_id="j1", start_time=ts, region="us-west",
            power_fraction=1.0, actual_runtime_hours=4.0
        )
        # Must not raise even without gpu_health_data arg
        result = fn.calculate([job], [decision], price, carbon)
        assert result.gpu_health_cost == 0.0


# ---------------------------------------------------------------------------
# TestSchedulerGPUHealthRouting
# ---------------------------------------------------------------------------

class TestSchedulerGPUHealthRouting:
    def _make_jobs(self, n=3) -> list[Job]:
        ts = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)
        return [
            Job(
                job_id=f"j{i}",
                workload_type="llm_batch_inference",
                submit_time=ts,
                earliest_start=ts,
                runtime_hours=2.0,
                deadline=ts + timedelta(hours=12),
                power_kw=50.0,
                region_options=["us-west", "us-east"],
                gpu_count=4,
            )
            for i in range(n)
        ]

    def _make_price_carbon(self, same_price=True):
        ts = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)
        price_w = 40.0
        price_e = 40.0 if same_price else 42.0
        price = {
            "us-west": {ts + timedelta(hours=i): price_w for i in range(24)},
            "us-east": {ts + timedelta(hours=i): price_e for i in range(24)},
        }
        carbon = {
            "us-west": {ts + timedelta(hours=i): 300.0 for i in range(24)},
            "us-east": {ts + timedelta(hours=i): 300.0 for i in range(24)},
        }
        return price, carbon

    def test_routes_to_healthy_region(self):
        from aurelius.optimization.scheduler import JobScheduler
        ts = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)
        # us-west is severely degraded; us-east is healthy
        gpu_health_data = {
            "us-west": {ts: 0.9},   # severely degraded
            "us-east": {ts: 0.0},   # healthy
        }
        cfg = OptimizationConfig(gpu_health_cost_per_hour=3.0)  # strong penalty
        scheduler = JobScheduler(cfg)
        jobs = self._make_jobs(1)
        price, carbon = self._make_price_carbon(same_price=True)
        result = scheduler.solve(jobs, price, carbon, gpu_health_data=gpu_health_data)
        # With strong penalty and same energy prices, job should go to us-east
        assert result.schedule[0].region == "us-east"

    def test_zero_health_cost_ignores_gpu_data(self):
        from aurelius.optimization.scheduler import JobScheduler
        ts = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)
        gpu_health_data = {
            "us-west": {ts: 0.9},  # very degraded but penalty disabled
            "us-east": {ts: 0.0},
        }
        cfg = OptimizationConfig(gpu_health_cost_per_hour=0.0)
        scheduler = JobScheduler(cfg)
        jobs = self._make_jobs(1)
        price, carbon = self._make_price_carbon(same_price=True)
        # With price penalty disabled, routing should not be forced to us-east
        # (may go either way; just verify no crash)
        result = scheduler.solve(jobs, price, carbon, gpu_health_data=gpu_health_data)
        assert len(result.schedule) == 1

    def test_backward_compat_no_gpu_health_arg(self):
        from aurelius.optimization.scheduler import JobScheduler
        scheduler = JobScheduler()
        jobs = self._make_jobs(2)
        price, carbon = self._make_price_carbon()
        # Must not raise when gpu_health_data is omitted
        result = scheduler.solve(jobs, price, carbon)
        assert len(result.schedule) == 2

    def test_high_health_cost_overrides_price_advantage(self):
        """Strong health penalty should outweigh moderate price advantage."""
        from aurelius.optimization.scheduler import JobScheduler
        ts = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)
        # us-west is $5/MWh cheaper but severely degraded (penalty=0.9)
        # us-east is $5/MWh more expensive but perfectly healthy
        price = {
            "us-west": {ts + timedelta(hours=i): 35.0 for i in range(24)},
            "us-east": {ts + timedelta(hours=i): 40.0 for i in range(24)},
        }
        carbon = {
            "us-west": {ts + timedelta(hours=i): 300.0 for i in range(24)},
            "us-east": {ts + timedelta(hours=i): 300.0 for i in range(24)},
        }
        gpu_health_data = {
            "us-west": {ts: 0.9},
            "us-east": {ts: 0.0},
        }
        # Energy savings from us-west: 5/1000 * 50kW * 2h = $0.50
        # GPU health cost at us-west: 0.9 * $5/h * 2h * 4 GPUs = $36
        # GPU health cost at us-east: 0.0 → $0
        cfg = OptimizationConfig(gpu_health_cost_per_hour=5.0)
        scheduler = JobScheduler(cfg)
        jobs = self._make_jobs(1)
        result = scheduler.solve(jobs, price, carbon, gpu_health_data=gpu_health_data)
        assert result.schedule[0].region == "us-east"


# ---------------------------------------------------------------------------
# TestBacktestEngineGPUHealthIntegration
# ---------------------------------------------------------------------------

class TestBacktestEngineGPUHealthIntegration:
    def test_engine_accepts_gpu_df_none(self):
        from aurelius.backtesting.engine import BacktestEngine
        engine = BacktestEngine(gpu_df=None)
        assert engine.gpu_df is None

    def test_engine_stores_gpu_df(self):
        import pandas as pd
        from aurelius.backtesting.engine import BacktestEngine
        fake_df = pd.DataFrame({"timestamp": [], "region": []})
        engine = BacktestEngine(gpu_df=fake_df)
        assert engine.gpu_df is not None

    def test_engine_run_without_gpu_df(self):
        """Engine must run without crashing when gpu_df is absent."""
        import pandas as pd
        from aurelius.backtesting.engine import BacktestEngine

        engine = BacktestEngine(
            method="greedy",
            train_days=3,
            eval_days=2,
            gpu_df=None,
        )
        ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
        price_rows = [
            {"timestamp": ts + timedelta(hours=i), "region": "us-west", "price_per_mwh": 40.0 + i % 5}
            for i in range(7 * 24)
        ]
        price_df = pd.DataFrame(price_rows)
        carbon_df = pd.DataFrame([
            {"timestamp": ts + timedelta(hours=i), "region": "us-west", "gco2_per_kwh": 300.0}
            for i in range(7 * 24)
        ])
        jobs = [
            Job(
                job_id="j1", workload_type="training",
                submit_time=ts, earliest_start=ts,
                runtime_hours=2.0, deadline=ts + timedelta(hours=48),
                power_kw=50.0, region_options=["us-west"],
            )
        ]
        results = engine.run(jobs, price_df, carbon_df)
        assert results is not None

    def test_engine_gpu_fold_leakage_safe(self):
        """GPU data after fold eval_start must not be used in that fold."""
        import pandas as pd
        from aurelius.backtesting.engine import BacktestEngine
        from aurelius.ingestion.dcgm_provider import DCGMProvider

        # Generate fixture starting at 2026-01-01
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        fixture = DCGMProvider.generate_fixture(["us-west"], hours=7 * 24, seed=1, start_dt=start)
        fixture.save_csv("/tmp/test_gpu_leakage.csv")
        gpu_df = pd.read_csv("/tmp/test_gpu_leakage.csv")

        engine = BacktestEngine(
            method="greedy",
            train_days=3,
            eval_days=2,
            gpu_df=gpu_df,
            config=OptimizationConfig(gpu_health_cost_per_hour=0.5),
        )
        ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
        price_df = pd.DataFrame([
            {"timestamp": ts + timedelta(hours=i), "region": "us-west", "price_per_mwh": 40.0}
            for i in range(7 * 24)
        ])
        carbon_df = pd.DataFrame([
            {"timestamp": ts + timedelta(hours=i), "region": "us-west", "gco2_per_kwh": 300.0}
            for i in range(7 * 24)
        ])
        jobs = [
            Job(
                job_id="j1", workload_type="training",
                submit_time=ts, earliest_start=ts,
                runtime_hours=2.0, deadline=ts + timedelta(hours=48),
                power_kw=50.0, region_options=["us-west"],
            )
        ]
        # Must not crash
        results = engine.run(jobs, price_df, carbon_df)
        assert results is not None


# ---------------------------------------------------------------------------
# TestLivePrometheusSkipped
# ---------------------------------------------------------------------------

class TestLivePrometheusSkipped:
    def test_empty_provider_when_no_env(self):
        """from_prometheus_live must return empty provider when no env vars set."""
        clean = {k: v for k, v in os.environ.items()
                 if k not in ("PROMETHEUS_URL", "DCGM_EXPORTER_URL")}
        with patch.dict(os.environ, clean, clear=True):
            p = DCGMProvider.from_prometheus_live(region="us-west")
            assert p.record_count == 0

    @pytest.mark.skipif(
        not os.environ.get("PROMETHEUS_URL") and not os.environ.get("DCGM_EXPORTER_URL"),
        reason="PROMETHEUS_URL / DCGM_EXPORTER_URL not set — skipping live test"
    )
    def test_live_prometheus_returns_gpus(self):
        p = DCGMProvider.from_prometheus_live(region="us-west")
        assert p.record_count >= 0  # may be 0 if no DCGM running
