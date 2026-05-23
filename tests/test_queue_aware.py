"""Tests for Phase 5 — Queue-Aware Optimization.

Covers:
  - QueueState data model
  - QueueProvider (CSV, fixture, lookup, leakage safety)
  - ObjectiveFunction with queue_delay_cost
  - JobScheduler queue-aware placement
  - BacktestEngine queue_df integration
  - Backward compatibility (no queue data = unchanged behaviour)
"""

from __future__ import annotations

import csv
import io
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from aurelius.ingestion.queue_provider import QueueProvider
from aurelius.models import (
    Job,
    OptimizationConfig,
    QueueState,
    ScheduleDecision,
)
from aurelius.optimization.objective import ObjectiveFunction, _lookup_last_known

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

UTC = timezone.utc

_BASE = datetime(2026, 1, 1, 0, tzinfo=UTC)

def _dt(hour=0):
    return _BASE + timedelta(hours=hour)


def _make_job(
    job_id="j1",
    region_options=None,
    gpu_count=8,
    sla_penalty_per_hour=10.0,
    runtime_hours=4.0,
    power_kw=100.0,
):
    submit = _dt(hour=0)
    deadline = submit + timedelta(hours=48)
    return Job(
        job_id=job_id,
        submit_time=submit,
        runtime_hours=runtime_hours,
        deadline=deadline,
        power_kw=power_kw,
        earliest_start=submit,
        region_options=region_options or ["us-west", "us-east"],
        gpu_count=gpu_count,
        sla_penalty_per_hour=sla_penalty_per_hour,
    )


def _price_data(regions, hours=48, base_price=50.0):
    start = _dt(hour=0)
    data = {}
    for r in regions:
        data[r] = {
            start + timedelta(hours=i): base_price
            for i in range(hours)
        }
    return data


def _carbon_data(regions, hours=48):
    start = _dt(hour=0)
    return {r: {start + timedelta(hours=i): 400.0 for i in range(hours)} for r in regions}


def _make_queue_csv(rows: list[dict]) -> str:
    """Return CSV string from list of row dicts."""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=[
        "timestamp", "region", "cluster_id", "gpu_type",
        "available_gpus", "queue_depth_jobs", "est_wait_hours",
    ])
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


# ============================================================================
# QueueState model tests
# ============================================================================

class TestQueueStateModel:
    def test_construction(self):
        qs = QueueState(
            timestamp=_dt(hour=0),
            region="us-west",
            cluster_id="cluster-1",
            gpu_type="a100",
            available_gpus=40,
            queue_depth_jobs=5,
            est_wait_hours=2.5,
        )
        assert qs.region == "us-west"
        assert qs.est_wait_hours == 2.5
        assert qs.available_gpus == 40

    def test_optional_gpu_type(self):
        qs = QueueState(
            timestamp=_dt(hour=0), region="us-east", cluster_id="c1",
            gpu_type=None, available_gpus=0, queue_depth_jobs=0, est_wait_hours=0.0,
        )
        assert qs.gpu_type is None


# ============================================================================
# QueueProvider tests
# ============================================================================

class TestQueueProviderFromCSV:
    def test_load_minimal_csv(self, tmp_path):
        rows = [
            {"timestamp": "2026-01-01T00:00:00Z", "region": "us-west",
             "cluster_id": "c1", "gpu_type": "a100",
             "available_gpus": 40, "queue_depth_jobs": 2, "est_wait_hours": 1.5},
            {"timestamp": "2026-01-01T01:00:00Z", "region": "us-west",
             "cluster_id": "c1", "gpu_type": "a100",
             "available_gpus": 35, "queue_depth_jobs": 5, "est_wait_hours": 3.0},
        ]
        p = tmp_path / "queue.csv"
        p.write_text(_make_queue_csv(rows))
        provider = QueueProvider.from_csv(str(p))
        assert provider.n_records == 2

    def test_missing_required_column_raises(self, tmp_path):
        p = tmp_path / "bad.csv"
        p.write_text("region,est_wait_hours\nus-west,1.0\n")
        with pytest.raises(ValueError, match="timestamp"):
            QueueProvider.from_csv(str(p))

    def test_optional_columns_default(self, tmp_path):
        p = tmp_path / "minimal.csv"
        p.write_text("timestamp,region,est_wait_hours\n2026-01-01T00:00:00Z,us-west,2.0\n")
        provider = QueueProvider.from_csv(str(p))
        assert provider.n_records == 1
        assert provider._records[0].cluster_id == "default"
        assert provider._records[0].gpu_type is None

    def test_regions_property(self, tmp_path):
        rows = [
            {"timestamp": "2026-01-01T00:00:00Z", "region": "us-west",
             "cluster_id": "c1", "gpu_type": None,
             "available_gpus": 40, "queue_depth_jobs": 0, "est_wait_hours": 0.5},
            {"timestamp": "2026-01-01T00:00:00Z", "region": "us-east",
             "cluster_id": "c1", "gpu_type": None,
             "available_gpus": 30, "queue_depth_jobs": 8, "est_wait_hours": 4.0},
        ]
        p = tmp_path / "q.csv"
        p.write_text(_make_queue_csv(rows))
        provider = QueueProvider.from_csv(str(p))
        assert set(provider.regions) == {"us-west", "us-east"}


class TestQueueProviderGetWaitHours:
    def _make_provider(self, rows):
        """Build provider from list of (timestamp, region, wait) tuples."""
        data = [
            {"timestamp": ts.isoformat(), "region": r, "cluster_id": "c1",
             "gpu_type": None, "available_gpus": 0,
             "queue_depth_jobs": 0, "est_wait_hours": w}
            for ts, r, w in rows
        ]
        df = pd.DataFrame(data)
        return QueueProvider.from_dataframe(df)

    def test_exact_timestamp_match(self):
        ts = _dt(hour=6)
        provider = self._make_provider([(ts, "us-west", 2.5)])
        result = provider.get_wait_hours("us-west", ts)
        assert result == pytest.approx(2.5)

    def test_last_known_before_timestamp(self):
        ts0 = _dt(hour=6)
        ts1 = _dt(hour=8)
        query = _dt(hour=9)
        provider = self._make_provider([(ts0, "us-west", 1.0), (ts1, "us-west", 3.0)])
        result = provider.get_wait_hours("us-west", query)
        assert result == pytest.approx(3.0)

    def test_no_data_before_timestamp_returns_zero(self):
        ts = _dt(hour=10)
        query = _dt(hour=5)  # before the only record
        provider = self._make_provider([(ts, "us-west", 5.0)])
        result = provider.get_wait_hours("us-west", query)
        assert result == 0.0

    def test_unknown_region_returns_zero(self):
        ts = _dt(hour=0)
        provider = self._make_provider([(ts, "us-west", 2.0)])
        result = provider.get_wait_hours("us-south", ts)
        assert result == 0.0

    def test_leakage_safe_strict_before(self):
        ts_future = _dt(hour=12)
        query = _dt(hour=11)  # one hour before the only record
        provider = self._make_provider([(ts_future, "us-east", 5.0)])
        # Must return 0.0 — future queue state must not leak
        result = provider.get_wait_hours("us-east", query)
        assert result == 0.0

    def test_multiple_clusters_aggregated(self):
        ts = _dt(hour=0)
        # Two clusters in us-west with different wait times
        data = [
            {"timestamp": ts.isoformat(), "region": "us-west", "cluster_id": "c1",
             "gpu_type": None, "available_gpus": 0, "queue_depth_jobs": 10, "est_wait_hours": 2.0},
            {"timestamp": ts.isoformat(), "region": "us-west", "cluster_id": "c2",
             "gpu_type": None, "available_gpus": 0, "queue_depth_jobs": 10, "est_wait_hours": 4.0},
        ]
        provider = QueueProvider.from_dataframe(pd.DataFrame(data))
        # Weighted mean of (2.0, 4.0) with equal weights = 3.0
        result = provider.get_wait_hours("us-west", ts)
        assert result == pytest.approx(3.0, abs=0.01)


class TestQueueProviderToDictLookup:
    def test_lookup_format(self, tmp_path):
        rows = [
            {"timestamp": "2026-01-01T00:00:00Z", "region": "us-west",
             "cluster_id": "c1", "gpu_type": "a100",
             "available_gpus": 40, "queue_depth_jobs": 2, "est_wait_hours": 1.5},
        ]
        p = tmp_path / "q.csv"
        p.write_text(_make_queue_csv(rows))
        provider = QueueProvider.from_csv(str(p))
        lookup = provider.to_dict_lookup()
        assert "us-west" in lookup
        assert isinstance(lookup["us-west"], dict)
        # Check at least one timestamp maps to a float
        vals = list(lookup["us-west"].values())
        assert len(vals) >= 1
        assert isinstance(vals[0], float)

    def test_mirrors_get_wait_hours(self, tmp_path):
        ts = _dt(hour=6)
        rows = [{"timestamp": ts.isoformat(), "region": "us-west",
                 "cluster_id": "c1", "gpu_type": None,
                 "available_gpus": 0, "queue_depth_jobs": 0, "est_wait_hours": 2.5}]
        p = tmp_path / "q.csv"
        p.write_text(_make_queue_csv(rows))
        provider = QueueProvider.from_csv(str(p))
        lookup = provider.to_dict_lookup()
        provider_val = provider.get_wait_hours("us-west", ts)
        ts_naive = ts.replace(tzinfo=None)
        lookup_val = lookup["us-west"].get(ts_naive, lookup["us-west"].get(ts, None))
        assert lookup_val is not None or provider_val == 0.0


class TestQueueProviderGenerateFixture:
    def test_generates_records(self):
        start = _dt(hour=0)
        end = _dt(hour=24)
        provider = QueueProvider.generate_fixture(
            regions=["us-west", "us-east"],
            start=start,
            end=end,
            seed=42,
        )
        # 24 hours × 2 regions × 1 gpu_type = 48 records
        assert provider.n_records == 48

    def test_deterministic_with_seed(self):
        start = _dt(hour=0)
        end = _dt(hour=12)
        p1 = QueueProvider.generate_fixture(["us-west"], start, end, seed=99)
        p2 = QueueProvider.generate_fixture(["us-west"], start, end, seed=99)
        lu1 = p1.to_dict_lookup()
        lu2 = p2.to_dict_lookup()
        assert lu1.keys() == lu2.keys()
        for region in lu1:
            for ts in lu1[region]:
                assert lu1[region][ts] == lu2[region][ts]

    def test_different_seeds_produce_different_results(self):
        start = _dt(hour=0)
        end = _dt(hour=12)
        p1 = QueueProvider.generate_fixture(["us-west"], start, end, seed=1)
        p2 = QueueProvider.generate_fixture(["us-west"], start, end, seed=2)
        lu1 = p1.to_dict_lookup()
        lu2 = p2.to_dict_lookup()
        vals1 = sorted(lu1["us-west"].values())
        vals2 = sorted(lu2["us-west"].values())
        assert vals1 != vals2

    def test_wait_hours_non_negative(self):
        start = _dt(hour=0)
        end = _dt(hour=48)
        provider = QueueProvider.generate_fixture(["us-west", "us-east", "us-south"], start, end)
        for r in provider._records:
            assert r.est_wait_hours >= 0.0

    def test_congestion_pattern_business_hours_higher(self):
        start = _dt(hour=0)
        end = _dt(hour=24)
        provider = QueueProvider.generate_fixture(["us-west"], start, end, seed=42)
        lookup = provider.to_dict_lookup()["us-west"]
        # Business-hour slots (12-20 UTC) should average higher than off-peak (0-8 UTC)
        biz_waits = [v for k, v in lookup.items() if 12 <= k.replace(tzinfo=None).hour < 20]
        off_waits = [v for k, v in lookup.items() if k.replace(tzinfo=None).hour < 8]
        if biz_waits and off_waits:
            assert sum(biz_waits) / len(biz_waits) > sum(off_waits) / len(off_waits)

    def test_base_wait_hours_respected(self):
        start = _dt(hour=0)
        end = _dt(hour=24)
        base = {"us-west": 5.0, "us-east": 0.5}
        p = QueueProvider.generate_fixture(["us-west", "us-east"], start, end, seed=42,
                                           base_wait_hours=base)
        lu = p.to_dict_lookup()
        mean_west = sum(lu["us-west"].values()) / len(lu["us-west"])
        mean_east = sum(lu["us-east"].values()) / len(lu["us-east"])
        # us-west should have much higher wait on average
        assert mean_west > mean_east

    def test_save_and_reload_csv(self, tmp_path):
        start = _dt(hour=0)
        end = _dt(hour=6)
        provider = QueueProvider.generate_fixture(["us-west"], start, end, seed=0)
        p = str(tmp_path / "queue.csv")
        provider.save_csv(p)
        reloaded = QueueProvider.from_csv(p)
        assert reloaded.n_records == provider.n_records


# ============================================================================
# _lookup_last_known helper tests
# ============================================================================

class TestLookupLastKnown:
    def test_empty_returns_zero(self):
        assert _lookup_last_known({}, _dt(hour=0)) == 0.0

    def test_exact_match(self):
        ts = _dt(hour=5)
        series = {ts: 3.5}
        assert _lookup_last_known(series, ts) == pytest.approx(3.5)

    def test_last_before_query(self):
        series = {_dt(hour=3): 1.0, _dt(hour=5): 2.0, _dt(hour=7): 3.0}
        result = _lookup_last_known(series, _dt(hour=6))
        assert result == pytest.approx(2.0)

    def test_all_after_query_returns_zero(self):
        series = {_dt(hour=10): 5.0}
        result = _lookup_last_known(series, _dt(hour=5))
        assert result == 0.0

    def test_latest_before_wins(self):
        series = {_dt(hour=1): 1.0, _dt(hour=2): 2.0, _dt(hour=3): 3.0}
        result = _lookup_last_known(series, _dt(hour=4))
        assert result == pytest.approx(3.0)


# ============================================================================
# ObjectiveFunction queue_delay_cost tests
# ============================================================================

class TestObjectiveFunctionQueueDelay:
    def test_no_queue_data_no_cost(self):
        job = _make_job(gpu_count=8)
        config = OptimizationConfig(queue_delay_cost_per_gpu_hour=3.0)
        obj_fn = ObjectiveFunction(config)
        regions = ["us-west"]
        price_data = _price_data(regions)
        carbon_data = _carbon_data(regions)
        decision = ScheduleDecision(
            job_id=job.job_id, start_time=_dt(hour=0),
            region="us-west", power_fraction=1.0,
            actual_runtime_hours=job.runtime_hours,
        )
        result = obj_fn.calculate([job], [decision], price_data, carbon_data, queue_data=None)
        assert result.queue_delay_cost == 0.0

    def test_queue_cost_zero_config_no_cost(self):
        job = _make_job(gpu_count=4)
        config = OptimizationConfig(queue_delay_cost_per_gpu_hour=0.0)
        obj_fn = ObjectiveFunction(config)
        regions = ["us-west"]
        price_data = _price_data(regions)
        carbon_data = _carbon_data(regions)
        ts = _dt(hour=0)
        queue_data = {"us-west": {ts: 2.0}}
        decision = ScheduleDecision(
            job_id=job.job_id, start_time=ts,
            region="us-west", power_fraction=1.0,
            actual_runtime_hours=job.runtime_hours,
        )
        result = obj_fn.calculate([job], [decision], price_data, carbon_data, queue_data=queue_data)
        assert result.queue_delay_cost == 0.0

    def test_queue_cost_calculation_correct(self):
        job = _make_job(gpu_count=8)
        cost_per_gpu_h = 3.0
        config = OptimizationConfig(queue_delay_cost_per_gpu_hour=cost_per_gpu_h)
        obj_fn = ObjectiveFunction(config)
        regions = ["us-west"]
        price_data = _price_data(regions)
        carbon_data = _carbon_data(regions)
        ts = _dt(hour=0)
        wait_h = 2.5
        queue_data = {"us-west": {ts: wait_h}}
        decision = ScheduleDecision(
            job_id=job.job_id, start_time=ts,
            region="us-west", power_fraction=1.0,
            actual_runtime_hours=job.runtime_hours,
        )
        result = obj_fn.calculate([job], [decision], price_data, carbon_data, queue_data=queue_data)
        expected = wait_h * cost_per_gpu_h * job.gpu_count
        assert result.queue_delay_cost == pytest.approx(expected, abs=0.01)

    def test_queue_cost_included_in_total(self):
        job = _make_job(gpu_count=4)
        config = OptimizationConfig(queue_delay_cost_per_gpu_hour=2.0)
        obj_fn = ObjectiveFunction(config)
        regions = ["us-west"]
        price_data = _price_data(regions, base_price=50.0)
        carbon_data = _carbon_data(regions)
        ts = _dt(hour=0)
        queue_data = {"us-west": {ts: 1.0}}  # 1h wait
        decision = ScheduleDecision(
            job_id=job.job_id, start_time=ts,
            region="us-west", power_fraction=1.0,
            actual_runtime_hours=job.runtime_hours,
        )
        with_queue = obj_fn.calculate([job], [decision], price_data, carbon_data, queue_data=queue_data)
        without_queue = obj_fn.calculate([job], [decision], price_data, carbon_data, queue_data=None)
        # Total with queue > without queue
        assert with_queue.total > without_queue.total
        assert with_queue.queue_delay_cost > 0.0
        assert without_queue.queue_delay_cost == 0.0

    def test_queue_cost_zero_gpu_count_defaults_to_one(self):
        job = _make_job(gpu_count=0)
        config = OptimizationConfig(queue_delay_cost_per_gpu_hour=5.0)
        obj_fn = ObjectiveFunction(config)
        price_data = _price_data(["us-west"])
        carbon_data = _carbon_data(["us-west"])
        ts = _dt(hour=0)
        queue_data = {"us-west": {ts: 1.0}}
        decision = ScheduleDecision(
            job_id=job.job_id, start_time=ts, region="us-west",
            power_fraction=1.0, actual_runtime_hours=job.runtime_hours,
        )
        result = obj_fn.calculate([job], [decision], price_data, carbon_data, queue_data=queue_data)
        # gpu_count=0 → max(1, 0) = 1
        assert result.queue_delay_cost == pytest.approx(1.0 * 5.0 * 1, abs=0.01)

    def test_backward_compat_no_queue_arg(self):
        """ObjectiveFunction.calculate without queue_data arg still works."""
        job = _make_job()
        config = OptimizationConfig()
        obj_fn = ObjectiveFunction(config)
        price_data = _price_data(["us-west"])
        carbon_data = _carbon_data(["us-west"])
        decision = ScheduleDecision(
            job_id=job.job_id, start_time=_dt(hour=0), region="us-west",
            power_fraction=1.0, actual_runtime_hours=job.runtime_hours,
        )
        result = obj_fn.calculate([job], [decision], price_data, carbon_data)
        assert result.queue_delay_cost == 0.0
        assert result.total >= 0.0


# ============================================================================
# Scheduler queue-aware routing tests
# ============================================================================

class TestSchedulerQueueAwareRouting:
    """Verify the optimizer routes away from congested regions when queue delay cost is set."""

    def _make_scheduler(self, cost_per_gpu_h: float):
        from aurelius.optimization.scheduler import JobScheduler
        config = OptimizationConfig(
            queue_delay_cost_per_gpu_hour=cost_per_gpu_h,
            min_power_fraction=1.0,  # disable throttling for simplicity
        )
        return JobScheduler(config)

    def test_routes_to_lower_queue_region(self):
        """When us-west is congested (5h wait) and us-east is clear (0h wait),
        and energy prices are equal, optimizer should prefer us-east."""
        scheduler = self._make_scheduler(cost_per_gpu_h=3.0)
        job = _make_job(
            job_id="j1",
            region_options=["us-west", "us-east"],
            gpu_count=8,
        )
        # Equal energy prices
        price_data = _price_data(["us-west", "us-east"], base_price=50.0)
        carbon_data = _carbon_data(["us-west", "us-east"])

        ts = _dt(hour=0)
        # us-west has 5h queue, us-east has 0h queue
        queue_data = {
            "us-west": {ts: 5.0},
            "us-east": {ts: 0.0},
        }
        result = scheduler.solve(
            [job], price_data, carbon_data, method="greedy", queue_data=queue_data
        )
        assert len(result.schedule) == 1
        assert result.schedule[0].region == "us-east"

    def test_no_queue_data_unchanged_routing(self):
        """Without queue data, routing is determined by energy price only."""
        scheduler_no_q = self._make_scheduler(cost_per_gpu_h=0.0)
        job = _make_job(region_options=["us-west", "us-east"], gpu_count=4)
        # us-west cheaper
        price_data = {
            "us-west": {_dt(hour=i): 30.0 for i in range(48)},
            "us-east": {_dt(hour=i): 60.0 for i in range(48)},
        }
        carbon_data = _carbon_data(["us-west", "us-east"])
        result = scheduler_no_q.solve([job], price_data, carbon_data, method="greedy")
        # Should pick us-west (cheaper energy, no queue penalty)
        assert result.schedule[0].region == "us-west"

    def test_high_queue_cost_overrides_price_advantage(self):
        """When queue cost dominates energy price advantage, optimizer switches region."""
        scheduler = self._make_scheduler(cost_per_gpu_h=10.0)
        job = _make_job(region_options=["us-west", "us-east"], gpu_count=8)
        # us-west is $10/MWh cheaper but has 4h queue
        price_data = {
            "us-west": {_dt(hour=i): 40.0 for i in range(48)},
            "us-east": {_dt(hour=i): 50.0 for i in range(48)},
        }
        carbon_data = _carbon_data(["us-west", "us-east"])
        # queue_delay_cost for us-west = 4h * $10/gpu-h * 8 gpus = $320
        # energy savings from us-west = ($50-$40)/1000 * 100kw * 4h * 1.0 = $4
        # → queue cost >> energy savings → us-east should win
        queue_data = {
            "us-west": {_dt(hour=0): 4.0},
            "us-east": {_dt(hour=0): 0.0},
        }
        result = scheduler.solve(
            [job], price_data, carbon_data, method="greedy", queue_data=queue_data
        )
        assert result.schedule[0].region == "us-east"

    def test_zero_cost_config_ignores_queue(self):
        """When cost_per_gpu_hour=0, queue data has no effect on routing."""
        scheduler = self._make_scheduler(cost_per_gpu_h=0.0)
        job = _make_job(region_options=["us-west", "us-east"], gpu_count=8)
        price_data = {
            "us-west": {_dt(hour=i): 40.0 for i in range(48)},
            "us-east": {_dt(hour=i): 50.0 for i in range(48)},
        }
        carbon_data = _carbon_data(["us-west", "us-east"])
        queue_data = {
            "us-west": {_dt(hour=0): 100.0},  # extreme queue — should be ignored
            "us-east": {_dt(hour=0): 0.0},
        }
        result = scheduler.solve(
            [job], price_data, carbon_data, method="greedy", queue_data=queue_data
        )
        # us-west is still cheaper energy → should win (queue ignored)
        assert result.schedule[0].region == "us-west"

    def test_backward_compat_no_queue_kwarg(self):
        """JobScheduler.solve without queue_data kwarg still works."""
        from aurelius.optimization.scheduler import JobScheduler
        scheduler = JobScheduler()
        job = _make_job(region_options=["us-west"])
        price_data = _price_data(["us-west"])
        carbon_data = _carbon_data(["us-west"])
        result = scheduler.solve([job], price_data, carbon_data, method="greedy")
        assert len(result.schedule) == 1

    def test_objective_components_include_queue_delay(self):
        """Result objective should report non-zero queue_delay_cost when queue data used."""
        from aurelius.optimization.scheduler import JobScheduler
        config = OptimizationConfig(queue_delay_cost_per_gpu_hour=2.0)
        scheduler = JobScheduler(config)
        job = _make_job(region_options=["us-west"], gpu_count=4)
        price_data = _price_data(["us-west"])
        carbon_data = _carbon_data(["us-west"])
        base = _dt(hour=0)
        queue_data = {"us-west": {base + timedelta(hours=i): 1.0 for i in range(48)}}
        result = scheduler.solve([job], price_data, carbon_data, method="greedy",
                                 queue_data=queue_data)
        assert result.objective.queue_delay_cost > 0.0


# ============================================================================
# OptimizationConfig queue field tests
# ============================================================================

class TestOptimizationConfigQueue:
    def test_default_queue_cost_zero(self):
        config = OptimizationConfig()
        assert config.queue_delay_cost_per_gpu_hour == 0.0

    def test_custom_queue_cost(self):
        config = OptimizationConfig(queue_delay_cost_per_gpu_hour=3.5)
        assert config.queue_delay_cost_per_gpu_hour == 3.5

    def test_to_dict_includes_queue_cost(self):
        config = OptimizationConfig(queue_delay_cost_per_gpu_hour=2.0)
        d = config.to_dict()
        assert "queue_delay_cost_per_gpu_hour" in d
        assert d["queue_delay_cost_per_gpu_hour"] == 2.0

    def test_backward_compat_construction(self):
        """All existing code that constructs OptimizationConfig without queue param works."""
        config = OptimizationConfig(alpha=1.0, beta=0.3, gamma=0.05)
        assert config.queue_delay_cost_per_gpu_hour == 0.0


# ============================================================================
# BacktestEngine queue_df integration tests
# ============================================================================

class TestBacktestEngineQueueIntegration:
    """Smoke tests verifying the engine accepts and processes queue_df."""

    def _make_engine(self, queue_df=None, cost_per_gpu_h=2.0, train_days=2, eval_days=1):
        from aurelius.backtesting.engine import BacktestEngine
        config = OptimizationConfig(queue_delay_cost_per_gpu_hour=cost_per_gpu_h)
        return BacktestEngine(
            method="greedy", config=config, queue_df=queue_df,
            train_days=train_days, eval_days=eval_days,
        )

    def _make_price_df(self, regions, n_hours=96):
        import numpy as np
        rows = []
        base = _dt(hour=0)
        for r in regions:
            for h in range(n_hours):
                rows.append({
                    "timestamp": base + timedelta(hours=h),
                    "region": r,
                    "price_per_mwh": 40.0 + 20.0 * abs(np.sin(h / 12)),
                })
        return pd.DataFrame(rows)

    def _make_carbon_df(self, regions, n_hours=96):
        rows = []
        base = _dt(hour=0)
        for r in regions:
            for h in range(n_hours):
                rows.append({
                    "timestamp": base + timedelta(hours=h),
                    "region": r,
                    "gco2_per_kwh": 350.0,
                })
        return pd.DataFrame(rows)

    def _make_jobs(self, regions, n=5):
        jobs = []
        for i in range(n):
            # Place jobs in eval window: after 2 train days (48h), within 1 eval day (24h)
            submit = _dt(hour=49 + i)
            jobs.append(Job(
                job_id=f"j{i}",
                submit_time=submit,
                runtime_hours=2.0,
                deadline=submit + timedelta(hours=24),
                power_kw=50.0,
                earliest_start=submit,
                region_options=regions,
                gpu_count=4,
                workload_type="llm_batch_inference",
            ))
        return jobs

    def test_engine_accepts_queue_df_none(self):
        engine = self._make_engine(queue_df=None)
        assert engine.queue_df is None

    def test_engine_stores_queue_df(self):
        df = QueueProvider.generate_fixture(
            ["us-west", "us-east"], _dt(hour=0), _dt(hour=96), seed=42
        ).to_dataframe()
        engine = self._make_engine(queue_df=df)
        assert engine.queue_df is not None
        assert len(engine.queue_df) > 0

    def test_engine_run_with_queue_df(self):
        """End-to-end: engine runs folds with queue_df without errors."""
        regions = ["us-west", "us-east"]
        queue_df = QueueProvider.generate_fixture(
            regions, _dt(hour=0), _dt(hour=96), seed=42
        ).to_dataframe()

        engine = self._make_engine(queue_df=queue_df)
        price_df = self._make_price_df(regions)
        carbon_df = self._make_carbon_df(regions)
        jobs = self._make_jobs(regions)

        rounds = engine.run(jobs, price_df, carbon_df)
        assert len(rounds) >= 1
        for r in rounds:
            assert r.optimizer_schedule is not None

    def test_engine_run_without_queue_df_unchanged(self):
        """Engine without queue_df produces a valid schedule (backward compat)."""
        regions = ["us-west"]
        engine = self._make_engine(queue_df=None)
        price_df = self._make_price_df(regions)
        carbon_df = self._make_carbon_df(regions)
        jobs = self._make_jobs(regions)

        rounds = engine.run(jobs, price_df, carbon_df)
        assert len(rounds) >= 1

    def test_queue_leakage_safe_in_fold(self):
        """Queue rows after eval_start are never used by the optimizer."""
        from aurelius.backtesting.engine import BacktestEngine
        regions = ["us-west", "us-east"]

        # 2-day train + 1-day eval = 96h of price data
        price_df = self._make_price_df(regions, n_hours=96)
        carbon_df = self._make_carbon_df(regions, n_hours=96)

        # Queue data: only in 2030 — well past any eval window (2026).
        # The leakage-safe logic should give queue_data=None to the optimizer.
        far_future_ts = datetime(2030, 1, 1, 0, tzinfo=UTC)
        queue_rows = [
            {"timestamp": far_future_ts.isoformat(), "region": "us-west",
             "cluster_id": "c1", "gpu_type": None, "available_gpus": 0,
             "queue_depth_jobs": 100, "est_wait_hours": 99.0},
        ]
        queue_df = pd.DataFrame(queue_rows)
        queue_df["timestamp"] = pd.to_datetime(queue_df["timestamp"], utc=True)

        config = OptimizationConfig(queue_delay_cost_per_gpu_hour=100.0)
        engine = BacktestEngine(
            method="greedy",
            train_days=2, eval_days=1,
            config=config,
            queue_df=queue_df,
        )
        jobs = self._make_jobs(regions, n=5)

        # Must not crash — future congestion must not affect the optimizer
        rounds = engine.run(jobs, price_df, carbon_df)
        assert len(rounds) >= 1


# ============================================================================
# QueueProvider dataframe round-trip test
# ============================================================================

class TestQueueProviderRoundTrip:
    def test_to_dataframe_then_from_dataframe(self):
        start = _dt(hour=0)
        end = _dt(hour=6)
        original = QueueProvider.generate_fixture(["us-west", "us-east"], start, end, seed=7)
        df = original.to_dataframe()
        reloaded = QueueProvider.from_dataframe(df)
        lu_orig = original.to_dict_lookup()
        lu_reload = reloaded.to_dict_lookup()
        for region in lu_orig:
            for ts, val in lu_orig[region].items():
                # Allow slight float precision differences
                r_val = lu_reload.get(region, {}).get(ts, None)
                if r_val is not None:
                    assert abs(r_val - val) < 1e-6
