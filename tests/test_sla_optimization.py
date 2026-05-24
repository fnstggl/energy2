"""Tests that SLAs actually change decisions in the real JobScheduler path.

Runs the optimizer in two modes — SLA disabled (unconstrained) vs SLA enabled —
and asserts the chosen region/migration differs when an SLA is relevant. Also
exercises the before/after SLA report.
"""

from datetime import datetime, timedelta, timezone

from aurelius.models import Job, OptimizationConfig
from aurelius.optimization.scheduler import JobScheduler
from aurelius.sla import (
    ActionType,
    OptimizationAction,
    RegionContext,
    SLAAwareActionSelector,
    SLALoader,
    SLAReport,
    WorkloadState,
)

UTC = timezone.utc
T0 = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)


def _hours(n):
    return timedelta(hours=n)


def _price_series(regions, hours=48):
    """Build {region: {ts: price}} with ercot much cheaper than us-east."""
    data = {}
    cheap = {"ercot": 10.0, "us-west": 40.0, "us-east": 80.0, "eu-west": 60.0}
    for r in regions:
        series = {}
        for h in range(hours):
            ts = T0 + _hours(h)
            series[ts] = cheap.get(r, 50.0)
        data[r] = series
    return data


def _carbon_series(regions, hours=48):
    return {r: {T0 + _hours(h): 300.0 for h in range(hours)} for r in regions}


def _make_job(regions):
    return Job(
        job_id="inference-prod",
        submit_time=T0,
        runtime_hours=4.0,
        deadline=T0 + _hours(24),
        power_kw=100.0,
        earliest_start=T0,
        region_options=list(regions),
        priority=5,
        workload_type="realtime_inference",
    )


class TestSchedulerSLAWiring:
    def test_unconstrained_picks_cheapest_region(self):
        regions = ["us-east", "ercot"]
        job = _make_job(regions)
        sched = JobScheduler(OptimizationConfig(default_region="us-east"))
        result = sched.solve(_make_job_list(job), _price_series(regions),
                             _carbon_series(regions), method="greedy")
        # Without SLA, optimizer chases the cheap region.
        assert result.schedule[0].region == "ercot"

    def test_sla_forbidden_region_blocks_cheapest(self):
        regions = ["us-east", "ercot"]
        job = _make_job(regions)
        text = """
        policies:
          - name: inf
            tier: latency_sensitive
            applies_to_workloads: [inference-prod]
            hard:
              allowed_regions: [us-east]
              max_p99_latency_ms: 3000
        """
        reg = SLALoader.load_text(text, fmt="yaml")
        rc = {
            "us-east": RegionContext(region="us-east", baseline_p99_latency_ms=500,
                                     spare_capacity_pct=60),
            "ercot": RegionContext(region="ercot", baseline_p99_latency_ms=500,
                                   spare_capacity_pct=60),
        }
        states = {"inference-prod": WorkloadState(region="us-east", p99_latency_ms=500,
                                                  availability_pct=99.99, capacity_buffer_pct=50,
                                                  queue_wait_ms=10)}
        sched = JobScheduler(
            OptimizationConfig(default_region="us-east"),
            sla_registry=reg, region_contexts=rc, current_states=states,
        )
        result = sched.solve(_make_job_list(job), _price_series(regions),
                             _carbon_series(regions), method="greedy")
        # SLA forbids ercot -> optimizer must keep us-east despite higher cost.
        assert result.schedule[0].region == "us-east"

    def test_sla_disabled_registry_preserves_legacy(self):
        regions = ["us-east", "ercot"]
        job = _make_job(regions)
        text = """
        enabled: false
        policies:
          - name: inf
            tier: critical
            applies_to_workloads: [inference-prod]
            hard:
              allowed_regions: [us-east]
        """
        reg = SLALoader.load_text(text, fmt="yaml")
        sched = JobScheduler(OptimizationConfig(default_region="us-east"), sla_registry=reg)
        result = sched.solve(_make_job_list(job), _price_series(regions),
                             _carbon_series(regions), method="greedy")
        # Registry disabled -> behaves like no SLA -> picks cheapest.
        assert result.schedule[0].region == "ercot"

    def test_no_sla_registry_is_unchanged(self):
        # Sanity: scheduler with no SLA args behaves exactly as before.
        regions = ["us-east", "ercot"]
        job = _make_job(regions)
        sched = JobScheduler(OptimizationConfig(default_region="us-east"))
        assert not sched.sla_enabled
        result = sched.solve(_make_job_list(job), _price_series(regions),
                             _carbon_series(regions), method="greedy")
        assert result.schedule[0].region == "ercot"

    def test_migration_suppressed_when_forbidden(self):
        # A long job that would otherwise migrate to chase cheap prices.
        regions = ["us-east", "ercot"]
        job = Job(
            job_id="train-job", submit_time=T0, runtime_hours=8.0,
            deadline=T0 + _hours(24), power_kw=100.0, earliest_start=T0,
            region_options=regions, priority=1, workload_type="training",
            migration_cost_hours=0.5,
        )
        text = """
        policies:
          - name: crit
            tier: critical
            applies_to_workloads: [train-job]
            hard:
              migration_allowed: false
        """
        reg = SLALoader.load_text(text, fmt="yaml")
        # Time-varying prices to make migration attractive.
        price = {r: {} for r in regions}
        for h in range(48):
            ts = T0 + _hours(h)
            price["us-east"][ts] = 10.0 if h < 4 else 90.0
            price["ercot"][ts] = 90.0 if h < 4 else 10.0
        sched = JobScheduler(
            OptimizationConfig(default_region="us-east"),
            sla_registry=reg,
            current_states={"train-job": WorkloadState(region="us-east")},
            region_contexts={r: RegionContext(region=r, spare_capacity_pct=60) for r in regions},
        )
        result = sched.solve(_make_job_list(job), price, _carbon_series(regions),
                             method="greedy_migrate")
        # migration_allowed=false -> no segments / no migration added.
        assert result.schedule[0].migration_count == 0


def _make_job_list(job):
    return [job]


class TestSLAReport:
    def test_before_after_report(self):
        sel = SLAAwareActionSelector()
        report = SLAReport()

        # Critical inference workload: cheap ercot migration blocked.
        text = """
        policies:
          - name: inf
            tier: critical
            applies_to_workloads: [inference-prod]
            hard:
              allowed_regions: [us-east, us-west]
              max_p99_latency_ms: 3000
              migration_allowed: false
        """
        reg = SLALoader.load_text(text, fmt="yaml")
        pol = reg.resolve(workload_id="inference-prod")

        class _W:
            job_id = "inference-prod"
            workload_type = "realtime_inference"

        cur = WorkloadState(region="us-east", p99_latency_ms=1500, availability_pct=99.99,
                            capacity_buffer_pct=50, queue_wait_ms=10)
        rc = {"ercot": RegionContext(region="ercot", baseline_p99_latency_ms=6000,
                                     spare_capacity_pct=10)}
        acts = [OptimizationAction(ActionType.MIGRATE, "ercot", expected_savings_pct=18.4)]
        dec = sel.select(_W(), acts, cur, pol, region_contexts=rc)
        report.add(dec)

        text2 = """
        policies:
          - name: batch
            tier: batch
            applies_to_workloads: [batch-job]
        """
        pol2 = SLALoader.load_text(text2, fmt="yaml").resolve(workload_id="batch-job")

        class _B:
            job_id = "batch-job"
            workload_type = "training"

        curb = WorkloadState(region="us-east", p99_latency_ms=2000, availability_pct=99.5,
                             capacity_buffer_pct=20, queue_wait_ms=1000)
        rcb = {"ercot": RegionContext(region="ercot", baseline_p99_latency_ms=3000,
                                      spare_capacity_pct=40, baseline_queue_wait_ms=2000)}
        actsb = [OptimizationAction(ActionType.MIGRATE, "ercot", expected_savings_pct=18.4)]
        decb = sel.select(_B(), actsb, curb, pol2, region_contexts=rcb)
        report.add(decb)

        rendered = report.render_text()
        assert "inference-prod" in rendered
        assert "Blocked because" in rendered
        assert "18.4" in rendered

        d = report.to_dict()
        assert d["workloads"] == 2
        assert d["corrected_count"] == 1  # only the inference workload corrected
        # batch job migrates freely (no hard violation).
        assert decb.chosen_action.target_region == "ercot"
        assert not decb.was_corrected
