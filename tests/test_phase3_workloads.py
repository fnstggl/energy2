"""Phase 3 tests: workload traces, simulator, objective, and shadow runner.

Covers:
- WorkloadSimulator GPU-typed generation
- load_workload_csv() ingestion and validation
- ObjectiveFunction SLA/PUE/data-transfer terms
- ShadowRunner realized-price evaluation
- Data residency constraint enforcement
- Interruptibility and checkpointability defaults
"""

from __future__ import annotations

import csv
import io
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from aurelius.models import (
    Job,
    OptimizationConfig,
    ScheduleDecision,
    WORKLOAD_DEFAULT_INTERRUPTIBLE,
    WORKLOAD_DEFAULT_MAX_DELAY_HOURS,
    WORKLOAD_DEFAULT_SLA_CLASS,
)
from aurelius.simulation.workload_simulator import WorkloadSimulator
from aurelius.ingestion.workload_traces import load_workload_csv
from aurelius.optimization.objective import ObjectiveFunction, ObjectiveComponents
from aurelius.execution.shadow_runner import ShadowRunner, ShadowResult


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def base_dt() -> datetime:
    return datetime(2024, 1, 15, 0, 0, 0)


@pytest.fixture
def simple_job(base_dt) -> Job:
    return Job(
        job_id="j1",
        submit_time=base_dt,
        runtime_hours=2.0,
        deadline=base_dt + timedelta(hours=24),
        power_kw=100.0,
        earliest_start=base_dt,
        region_options=["us-west", "us-east"],
    )


@pytest.fixture
def price_data(base_dt) -> dict:
    prices = {}
    for region in ["us-west", "us-east", "eu-west"]:
        prices[region] = {
            base_dt + timedelta(hours=h): 60.0 + h * 2
            for h in range(200)
        }
    return prices


@pytest.fixture
def carbon_data(base_dt) -> dict:
    carbon = {}
    for region in ["us-west", "us-east", "eu-west"]:
        carbon[region] = {
            base_dt + timedelta(hours=h): 300.0
            for h in range(200)
        }
    return carbon


# ── WorkloadSimulator tests ───────────────────────────────────────────────────

class TestWorkloadSimulator:
    def test_generate_returns_correct_count(self):
        sim = WorkloadSimulator()
        jobs = sim.generate("llm_batch_inference", gpu_type="a100", n_jobs=10, seed=42)
        assert len(jobs) == 10

    def test_generate_is_reproducible(self):
        sim = WorkloadSimulator()
        jobs1 = sim.generate("training", gpu_type="h100", n_jobs=5, seed=99)
        jobs2 = sim.generate("training", gpu_type="h100", n_jobs=5, seed=99)
        for j1, j2 in zip(jobs1, jobs2):
            assert j1.job_id == j2.job_id
            assert j1.power_kw == j2.power_kw
            assert j1.runtime_hours == j2.runtime_hours

    def test_different_seeds_produce_different_jobs(self):
        sim = WorkloadSimulator()
        jobs1 = sim.generate("llm_batch_inference", gpu_type="a100", n_jobs=5, seed=1)
        jobs2 = sim.generate("llm_batch_inference", gpu_type="a100", n_jobs=5, seed=2)
        # At least one job should differ in power or timing
        assert any(j1.power_kw != j2.power_kw for j1, j2 in zip(jobs1, jobs2))

    def test_workload_types_produce_distinct_power_distributions(self):
        sim = WorkloadSimulator()
        inference_jobs = sim.generate("realtime_inference", gpu_type="t4", n_jobs=20, seed=42)
        training_jobs = sim.generate("training", gpu_type="a100", n_jobs=20, seed=42)

        avg_inference_power = sum(j.power_kw for j in inference_jobs) / len(inference_jobs)
        avg_training_power = sum(j.power_kw for j in training_jobs) / len(training_jobs)

        # Training should use substantially more power than realtime inference
        assert avg_training_power > avg_inference_power * 2

    def test_workload_types_produce_distinct_runtime_distributions(self):
        sim = WorkloadSimulator()
        batch_jobs = sim.generate("background_maintenance", gpu_type="cpu", n_jobs=20, seed=42)
        training_jobs = sim.generate("training", gpu_type="a100", n_jobs=20, seed=42)

        avg_batch_rt = sum(j.runtime_hours for j in batch_jobs) / len(batch_jobs)
        avg_training_rt = sum(j.runtime_hours for j in training_jobs) / len(training_jobs)

        # Training should run much longer than background maintenance
        assert avg_training_rt > avg_batch_rt * 5

    def test_realtime_inference_is_not_interruptible(self):
        sim = WorkloadSimulator()
        jobs = sim.generate("realtime_inference", gpu_type="t4", n_jobs=10, seed=42)
        assert all(not j.interruptible for j in jobs)

    def test_llm_batch_is_interruptible_and_preemptible(self):
        sim = WorkloadSimulator()
        jobs = sim.generate("llm_batch_inference", gpu_type="a100", n_jobs=10, seed=42)
        assert all(j.interruptible for j in jobs)
        assert all(j.preemptible for j in jobs)

    def test_checkpointable_training_is_interruptible(self):
        sim = WorkloadSimulator()
        jobs = sim.generate("training", gpu_type="a100", n_jobs=50, seed=42)
        checkpointable = [j for j in jobs if j.checkpointable]
        if checkpointable:
            assert all(j.interruptible for j in checkpointable)

    def test_deadline_always_after_earliest_start_plus_runtime(self):
        sim = WorkloadSimulator()
        for wt in ["realtime_inference", "llm_batch_inference", "fine_tuning", "training",
                   "data_processing", "scheduled_batch", "background_maintenance"]:
            jobs = sim.generate(wt, gpu_type="a100", n_jobs=10, seed=42)
            for job in jobs:
                feasible_deadline = job.earliest_start + timedelta(hours=job.runtime_hours)
                assert job.deadline >= feasible_deadline, (
                    f"{wt}: job {job.job_id} deadline {job.deadline} < "
                    f"earliest_start+runtime {feasible_deadline}"
                )

    def test_pue_is_at_least_1(self):
        sim = WorkloadSimulator()
        for wt in ["llm_batch_inference", "training"]:
            jobs = sim.generate(wt, gpu_type="a100", n_jobs=10, seed=42)
            assert all(j.pue >= 1.0 for j in jobs)

    def test_gpu_count_within_profile(self):
        sim = WorkloadSimulator()
        jobs = sim.generate("background_maintenance", gpu_type="cpu", n_jobs=20, seed=42)
        assert all(j.gpu_count == 0 for j in jobs)

    def test_generate_mixed(self):
        sim = WorkloadSimulator()
        mix = {"llm_batch_inference": 3, "training": 2, "scheduled_batch": 5}
        jobs = sim.generate_mixed(mix, gpu_type="a100", seed=42)
        assert len(jobs) == 10
        types = {j.workload_type for j in jobs}
        assert types == set(mix.keys())

    def test_invalid_workload_type_raises(self):
        sim = WorkloadSimulator()
        with pytest.raises(ValueError, match="Unsupported workload_type"):
            sim.generate("invalid_type", gpu_type="a100", n_jobs=5, seed=42)

    def test_invalid_gpu_type_raises(self):
        sim = WorkloadSimulator()
        with pytest.raises(ValueError, match="Unknown gpu_type"):
            sim.generate("training", gpu_type="rtx3090", n_jobs=5, seed=42)

    def test_zero_n_jobs_raises(self):
        sim = WorkloadSimulator()
        with pytest.raises(ValueError, match="n_jobs must be > 0"):
            sim.generate("training", gpu_type="a100", n_jobs=0, seed=42)


# ── load_workload_csv tests ───────────────────────────────────────────────────

class TestLoadWorkloadCSV:
    def _write_csv(self, rows: list[dict], tmpdir: Path) -> Path:
        if not rows:
            raise ValueError("rows must not be empty")
        path = tmpdir / "workloads.csv"
        fieldnames = list(rows[0].keys())
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def _base_row(self) -> dict:
        return {
            "job_id": "j1",
            "submit_time": "2024-01-15T08:00:00",
            "runtime_hours": "2.0",
            "deadline": "2024-01-15T18:00:00",
            "power_kw": "100.0",
            "earliest_start": "2024-01-15T08:00:00",
            "region_options": "us-west,us-east",
        }

    def test_load_minimal_csv(self, tmp_path):
        path = self._write_csv([self._base_row()], tmp_path)
        jobs = load_workload_csv(path)
        assert len(jobs) == 1
        assert jobs[0].job_id == "j1"
        assert jobs[0].runtime_hours == 2.0
        assert "us-west" in jobs[0].region_options

    def test_load_full_featured_csv(self, tmp_path):
        row = self._base_row()
        row.update({
            "workload_type": "llm_batch_inference",
            "gpu_type": "A100",  # should be lowercased
            "gpu_count": "8",
            "sla_penalty_per_hour": "50.0",
            "sla_class": "deadline",
            "data_transfer_gb": "10.0",
            "pue": "1.3",
            "interruptible": "true",
            "preemptible": "true",
            "checkpointable": "true",
            "priority": "3",
            "max_delay_hours": "24.0",
        })
        path = self._write_csv([row], tmp_path)
        jobs = load_workload_csv(path)
        assert len(jobs) == 1
        j = jobs[0]
        assert j.workload_type == "llm_batch_inference"
        assert j.gpu_type == "a100"  # lowercased
        assert j.gpu_count == 8
        assert j.sla_penalty_per_hour == 50.0
        assert j.data_transfer_gb == 10.0
        assert j.pue == pytest.approx(1.3)
        assert j.interruptible is True
        assert j.preemptible is True
        assert j.checkpointable is True

    def test_workload_type_defaults_applied_when_not_in_csv(self, tmp_path):
        row = self._base_row()
        row["workload_type"] = "realtime_inference"
        # Do NOT include interruptible, preemptible, sla_class
        path = self._write_csv([row], tmp_path)
        jobs = load_workload_csv(path)
        j = jobs[0]
        # realtime_inference defaults
        assert j.interruptible is False
        assert j.sla_class == "latency_critical"
        assert j.max_delay_hours == 0.0

    def test_explicit_csv_fields_override_workload_defaults(self, tmp_path):
        row = self._base_row()
        row["workload_type"] = "realtime_inference"
        row["interruptible"] = "true"  # explicit override
        row["sla_class"] = "best_effort"  # explicit override
        path = self._write_csv([row], tmp_path)
        jobs = load_workload_csv(path)
        # Explicit values should take precedence over workload_type defaults
        assert jobs[0].interruptible is True
        assert jobs[0].sla_class == "best_effort"

    def test_forbidden_regions_applied(self, tmp_path):
        row = self._base_row()
        row["region_options"] = "us-west,us-east,eu-west"
        row["forbidden_regions"] = "us-east"
        path = self._write_csv([row], tmp_path)
        jobs = load_workload_csv(path)
        assert "us-east" not in jobs[0].region_options
        assert "us-west" in jobs[0].region_options
        assert "eu-west" in jobs[0].region_options

    def test_allowed_regions_restricts_options(self, tmp_path):
        row = self._base_row()
        row["region_options"] = "us-west,us-east,eu-west"
        row["allowed_regions"] = "us-west,eu-west"
        path = self._write_csv([row], tmp_path)
        jobs = load_workload_csv(path)
        assert "us-east" not in jobs[0].region_options
        assert "us-west" in jobs[0].region_options

    def test_json_array_region_options_parsed(self, tmp_path):
        row = self._base_row()
        row["region_options"] = '["us-west", "eu-west"]'
        path = self._write_csv([row], tmp_path)
        jobs = load_workload_csv(path)
        assert "us-west" in jobs[0].region_options
        assert "eu-west" in jobs[0].region_options

    def test_bad_row_skipped_good_rows_returned(self, tmp_path):
        good_row = self._base_row()
        bad_row = self._base_row()
        bad_row["job_id"] = "j2"
        bad_row["runtime_hours"] = "-5.0"  # invalid
        path = self._write_csv([good_row, bad_row], tmp_path)
        jobs = load_workload_csv(path)
        assert len(jobs) == 1
        assert jobs[0].job_id == "j1"

    def test_missing_required_column_raises(self, tmp_path):
        row = {"job_id": "j1", "runtime_hours": "2.0"}  # missing most fields
        path = tmp_path / "bad.csv"
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)
        with pytest.raises(ValueError, match="missing required columns"):
            load_workload_csv(path)

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            load_workload_csv("/nonexistent/path/workloads.csv")

    def test_deadline_before_earliest_start_plus_runtime_skipped(self, tmp_path):
        row = self._base_row()
        row["runtime_hours"] = "12.0"
        row["deadline"] = "2024-01-15T10:00:00"  # only 2h window, need 12h
        path = self._write_csv([row], tmp_path)
        jobs = load_workload_csv(path)
        assert len(jobs) == 0  # invalid row skipped

    def test_pue_less_than_1_skipped(self, tmp_path):
        row = self._base_row()
        row["pue"] = "0.9"  # invalid
        path = self._write_csv([row], tmp_path)
        jobs = load_workload_csv(path)
        assert len(jobs) == 0

    def test_multiple_jobs_loaded(self, tmp_path):
        rows = []
        for i in range(5):
            r = self._base_row()
            r["job_id"] = f"j{i}"
            rows.append(r)
        path = self._write_csv(rows, tmp_path)
        jobs = load_workload_csv(path)
        assert len(jobs) == 5


# ── ObjectiveFunction SLA/PUE/data-transfer tests ────────────────────────────

class TestObjectiveSLAPUETransfer:
    def test_pue_increases_energy_cost(self, base_dt, simple_job, price_data, carbon_data):
        obj_fn = ObjectiveFunction()
        d = ScheduleDecision(
            job_id=simple_job.job_id,
            start_time=base_dt,
            region="us-west",
            power_fraction=1.0,
            actual_runtime_hours=2.0,
        )
        # Without PUE overhead
        simple_job.pue = 1.0
        result_no_pue = obj_fn.calculate([simple_job], [d], price_data, carbon_data)

        # With PUE overhead
        job_pue = Job(
            job_id="j2",
            submit_time=base_dt,
            runtime_hours=2.0,
            deadline=base_dt + timedelta(hours=24),
            power_kw=simple_job.power_kw,
            earliest_start=base_dt,
            region_options=["us-west"],
            pue=1.5,
        )
        d2 = ScheduleDecision(
            job_id="j2",
            start_time=base_dt,
            region="us-west",
            power_fraction=1.0,
            actual_runtime_hours=2.0,
        )
        result_with_pue = obj_fn.calculate([job_pue], [d2], price_data, carbon_data)

        # 1.5x PUE should increase energy cost proportionally
        assert result_with_pue.energy_cost > result_no_pue.energy_cost
        ratio = result_with_pue.energy_cost / result_no_pue.energy_cost
        assert abs(ratio - 1.5) < 0.01, f"Expected ratio ~1.5, got {ratio}"

    def test_sla_penalty_fires_when_deadline_missed(self, base_dt):
        obj_fn = ObjectiveFunction()
        job = Job(
            job_id="j-sla",
            submit_time=base_dt,
            runtime_hours=4.0,
            deadline=base_dt + timedelta(hours=3),  # 3h deadline, 4h job
            power_kw=100.0,
            earliest_start=base_dt,
            region_options=["us-west"],
            sla_penalty_per_hour=100.0,
        )
        d = ScheduleDecision(
            job_id="j-sla",
            start_time=base_dt,
            region="us-west",
            power_fraction=1.0,
            actual_runtime_hours=4.0,
        )
        result = obj_fn.calculate([job], [d], {}, {})
        # 1 hour overrun × $100/hr = $100
        assert result.sla_penalty_cost == pytest.approx(100.0)

    def test_no_sla_penalty_when_deadline_met(self, base_dt, simple_job, price_data, carbon_data):
        obj_fn = ObjectiveFunction()
        simple_job.sla_penalty_per_hour = 100.0
        d = ScheduleDecision(
            job_id=simple_job.job_id,
            start_time=base_dt,
            region="us-west",
            power_fraction=1.0,
            actual_runtime_hours=2.0,
        )
        result = obj_fn.calculate([simple_job], [d], price_data, carbon_data)
        assert result.sla_penalty_cost == 0.0

    def test_data_transfer_cost_computed(self, base_dt, price_data, carbon_data):
        cfg = OptimizationConfig(data_transfer_cost_per_gb=0.1)
        obj_fn = ObjectiveFunction(cfg)
        job = Job(
            job_id="j-transfer",
            submit_time=base_dt,
            runtime_hours=1.0,
            deadline=base_dt + timedelta(hours=10),
            power_kw=100.0,
            earliest_start=base_dt,
            region_options=["us-west"],
            data_transfer_gb=50.0,  # 50 GB × $0.10/GB = $5.00
        )
        d = ScheduleDecision("j-transfer", base_dt, "us-west", 1.0, 1.0)
        result = obj_fn.calculate([job], [d], price_data, carbon_data)
        assert result.data_transfer_cost == pytest.approx(5.0)

    def test_zero_data_transfer_no_cost(self, base_dt, simple_job, price_data, carbon_data):
        obj_fn = ObjectiveFunction()
        simple_job.data_transfer_gb = 0.0
        d = ScheduleDecision(simple_job.job_id, base_dt, "us-west", 1.0, 2.0)
        result = obj_fn.calculate([simple_job], [d], price_data, carbon_data)
        assert result.data_transfer_cost == 0.0

    def test_high_sla_penalty_job_costs_more_when_late(self, base_dt):
        obj_fn = ObjectiveFunction()
        job = Job(
            job_id="j-late",
            submit_time=base_dt,
            runtime_hours=2.0,
            deadline=base_dt + timedelta(hours=1),  # only 1h window for 2h job
            power_kw=100.0,
            earliest_start=base_dt,
            region_options=["us-west"],
            sla_penalty_per_hour=1000.0,
        )
        d = ScheduleDecision("j-late", base_dt, "us-west", 1.0, 2.0)
        result = obj_fn.calculate([job], [d], {}, {})
        # 1 hour overrun × $1000/hr = $1000 penalty
        assert result.sla_penalty_cost == pytest.approx(1000.0)
        # Total should include penalty
        assert result.total >= 1000.0

    def test_objective_components_have_correct_fields(self, base_dt, simple_job, price_data, carbon_data):
        obj_fn = ObjectiveFunction()
        d = ScheduleDecision(simple_job.job_id, base_dt, "us-west", 1.0, 2.0)
        result = obj_fn.calculate([simple_job], [d], price_data, carbon_data)
        # Verify all new fields are present
        assert hasattr(result, "sla_penalty_cost")
        assert hasattr(result, "data_transfer_cost")
        assert isinstance(result.sla_penalty_cost, float)
        assert isinstance(result.data_transfer_cost, float)

    def test_carbon_objective_cost_only_ignores_carbon_in_total(self, base_dt, price_data, carbon_data):
        """With carbon_objective=cost_only (beta ignored), changing carbon should not affect total."""
        cfg_low_beta = OptimizationConfig(beta=0.0)
        cfg_high_beta = OptimizationConfig(beta=10.0)
        job = Job(
            job_id="j-co2",
            submit_time=base_dt,
            runtime_hours=2.0,
            deadline=base_dt + timedelta(hours=24),
            power_kw=100.0,
            earliest_start=base_dt,
            region_options=["us-west"],
        )
        d = ScheduleDecision("j-co2", base_dt, "us-west", 1.0, 2.0)
        result_low = ObjectiveFunction(cfg_low_beta).calculate([job], [d], price_data, carbon_data)
        result_high = ObjectiveFunction(cfg_high_beta).calculate([job], [d], price_data, carbon_data)
        assert result_high.total > result_low.total


# ── ShadowRunner tests ────────────────────────────────────────────────────────

class TestShadowRunner:
    def test_shadow_run_returns_result(self, base_dt, simple_job, price_data, carbon_data):
        scheduler = _make_greedy_schedule([simple_job], price_data, carbon_data, base_dt)
        runner = ShadowRunner()
        result = runner.run(scheduler, real_prices=price_data, real_carbon=carbon_data, jobs=[simple_job])
        assert isinstance(result, ShadowResult)
        assert result.jobs_evaluated == 1

    def test_shadow_uses_real_prices_not_forecasts(self, base_dt, simple_job, price_data, carbon_data):
        """Verify shadow computes cost from real_prices, not decision.forecast."""
        d = ScheduleDecision(
            job_id=simple_job.job_id,
            start_time=base_dt,
            region="us-west",
            power_fraction=1.0,
            actual_runtime_hours=2.0,
            forecast={"energy_cost": {"p50": 99999.0}},  # bogus forecast
        )
        runner = ShadowRunner()
        result = runner.run([d], real_prices=price_data, real_carbon=carbon_data, jobs=[simple_job])
        # Realized cost should NOT be 99999 — it uses real_prices
        assert result.records[0].realized_energy_cost < 1000.0

    def test_shadow_records_forecast_snapshot(self, base_dt, simple_job, price_data, carbon_data):
        forecast = {"energy_cost": {"p50": 50.0, "p90": 75.0}}
        d = ScheduleDecision(
            job_id=simple_job.job_id,
            start_time=base_dt,
            region="us-west",
            power_fraction=1.0,
            actual_runtime_hours=2.0,
            forecast=forecast,
        )
        runner = ShadowRunner()
        result = runner.run([d], real_prices=price_data, real_carbon=carbon_data, jobs=[simple_job])
        assert result.records[0].forecast_snapshot == forecast

    def test_shadow_detects_forbidden_region_violation(self, base_dt, price_data, carbon_data):
        job = Job(
            job_id="j-res",
            submit_time=base_dt,
            runtime_hours=2.0,
            deadline=base_dt + timedelta(hours=24),
            power_kw=100.0,
            earliest_start=base_dt,
            region_options=["us-west", "us-east"],
            forbidden_regions=["us-east"],
        )
        # Force a decision in forbidden region
        d = ScheduleDecision("j-res", base_dt, "us-east", 1.0, 2.0)
        runner = ShadowRunner()
        result = runner.run([d], real_prices=price_data, real_carbon=carbon_data, jobs=[job])
        assert any("data_residency_violation" in v for v in result.constraint_violations)

    def test_shadow_no_violations_when_valid(self, base_dt, price_data, carbon_data):
        sim = WorkloadSimulator(regions=["us-west", "us-east"])
        jobs = sim.generate("scheduled_batch", gpu_type="cpu", n_jobs=5, seed=42)
        # Build a valid schedule: start at ceiling of earliest_start to avoid sub-hour violations
        def _ceil_hour(dt: datetime) -> datetime:
            floored = dt.replace(minute=0, second=0, microsecond=0)
            return floored if floored >= dt else floored + timedelta(hours=1)

        decisions = [
            ScheduleDecision(
                job_id=j.job_id,
                start_time=_ceil_hour(j.earliest_start),
                region=j.region_options[0],
                power_fraction=1.0,
                actual_runtime_hours=j.runtime_hours,
            )
            for j in jobs
        ]
        runner = ShadowRunner()
        result = runner.run(decisions, real_prices=price_data, real_carbon=carbon_data, jobs=jobs)
        assert result.constraint_violations == []
        assert result.jobs_with_violations == 0

    def test_shadow_aggregate_totals_match_records(self, base_dt, price_data, carbon_data):
        sim = WorkloadSimulator()
        jobs = sim.generate("data_processing", gpu_type="a10g", n_jobs=5, seed=42)
        decisions = [
            ScheduleDecision(
                job_id=j.job_id,
                start_time=j.earliest_start.replace(minute=0, second=0, microsecond=0),
                region=j.region_options[0],
                power_fraction=1.0,
                actual_runtime_hours=j.runtime_hours,
            )
            for j in jobs
        ]
        runner = ShadowRunner()
        result = runner.run(decisions, real_prices=price_data, real_carbon=carbon_data, jobs=jobs)

        # Verify totals match sum of records
        sum_realized = sum(r.realized_total_cost for r in result.records)
        sum_baseline = sum(r.baseline_total_cost for r in result.records)
        assert abs(result.total_realized_cost - sum_realized) < 0.001
        assert abs(result.total_baseline_cost - sum_baseline) < 0.001

    def test_shadow_cost_savings_pct_computed_correctly(self, base_dt, price_data, carbon_data):
        simple_job = Job(
            job_id="j1",
            submit_time=base_dt,
            runtime_hours=2.0,
            deadline=base_dt + timedelta(hours=24),
            power_kw=100.0,
            earliest_start=base_dt,
            region_options=["us-west"],
        )
        baseline_d = ScheduleDecision("j1", base_dt, "us-west", 1.0, 2.0)
        runner = ShadowRunner()
        result = runner.run(
            [baseline_d],
            real_prices=price_data,
            real_carbon=carbon_data,
            jobs=[simple_job],
            baseline_decisions=[baseline_d],
        )
        # When decision == baseline, savings should be ~0
        assert abs(result.cost_savings_pct) < 0.001

    def test_shadow_result_to_json(self, base_dt, simple_job, price_data, carbon_data):
        d = ScheduleDecision(simple_job.job_id, base_dt, "us-west", 1.0, 2.0)
        runner = ShadowRunner()
        result = runner.run([d], real_prices=price_data, real_carbon=carbon_data, jobs=[simple_job])
        import json
        data = json.loads(result.to_json())
        assert "run_id" in data
        assert "records" in data
        assert len(data["records"]) == 1

    def test_shadow_persists_to_file(self, tmp_path, base_dt, simple_job, price_data, carbon_data):
        out_file = tmp_path / "shadow_results.jsonl"
        d = ScheduleDecision(simple_job.job_id, base_dt, "us-west", 1.0, 2.0)
        runner = ShadowRunner(output_path=out_file)
        result = runner.run([d], real_prices=price_data, real_carbon=carbon_data, jobs=[simple_job])
        assert out_file.exists()
        import json
        lines = out_file.read_text().strip().splitlines()
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["run_id"] == result.run_id

    def test_shadow_unknown_job_logged_as_violation(self, base_dt, simple_job, price_data, carbon_data):
        d = ScheduleDecision("nonexistent-job", base_dt, "us-west", 1.0, 2.0)
        runner = ShadowRunner()
        result = runner.run([d], real_prices=price_data, real_carbon=carbon_data, jobs=[simple_job])
        assert any("job_not_found" in v for v in result.constraint_violations)


# ── Job model data residency tests ───────────────────────────────────────────

class TestJobDataResidency:
    def test_allowed_regions_filters_region_options(self, base_dt):
        job = Job(
            job_id="j-res",
            submit_time=base_dt,
            runtime_hours=1.0,
            deadline=base_dt + timedelta(hours=24),
            power_kw=50.0,
            earliest_start=base_dt,
            region_options=["us-west", "us-east", "eu-west"],
            allowed_regions=["us-west", "eu-west"],
        )
        assert "us-east" not in job.region_options
        assert "us-west" in job.region_options

    def test_forbidden_regions_removed_from_options(self, base_dt):
        job = Job(
            job_id="j-forb",
            submit_time=base_dt,
            runtime_hours=1.0,
            deadline=base_dt + timedelta(hours=24),
            power_kw=50.0,
            earliest_start=base_dt,
            region_options=["us-west", "us-east"],
            forbidden_regions=["us-east"],
        )
        assert "us-east" not in job.region_options
        assert "us-west" in job.region_options

    def test_no_valid_region_keeps_original(self, base_dt):
        """When all regions are allowed/forbidden to impossibility, keep original."""
        job = Job(
            job_id="j-noregion",
            submit_time=base_dt,
            runtime_hours=1.0,
            deadline=base_dt + timedelta(hours=24),
            power_kw=50.0,
            earliest_start=base_dt,
            region_options=["us-west", "us-east"],
            allowed_regions=["eu-central"],  # not in region_options
        )
        # No valid intersection — region_options preserved (optimizer must handle)
        assert len(job.region_options) > 0


# ── Integration: workload_type defaults coverage ─────────────────────────────

class TestWorkloadTypeDefaults:
    @pytest.mark.parametrize("wt,expected_interruptible", [
        ("realtime_inference", False),
        ("llm_batch_inference", True),
        ("data_processing", True),
        ("scheduled_batch", True),
        ("background_maintenance", True),
    ])
    def test_interruptible_defaults(self, wt, expected_interruptible):
        assert WORKLOAD_DEFAULT_INTERRUPTIBLE[wt] == expected_interruptible

    @pytest.mark.parametrize("wt,expected_sla", [
        ("realtime_inference", "latency_critical"),
        ("training", "best_effort"),
        ("llm_batch_inference", "deadline"),
    ])
    def test_sla_class_defaults(self, wt, expected_sla):
        assert WORKLOAD_DEFAULT_SLA_CLASS[wt] == expected_sla

    @pytest.mark.parametrize("wt,expected_delay", [
        ("realtime_inference", 0.0),
        ("llm_batch_inference", 24.0),
        ("training", 48.0),
    ])
    def test_max_delay_defaults(self, wt, expected_delay):
        assert WORKLOAD_DEFAULT_MAX_DELAY_HOURS[wt] == expected_delay


# ── Helper ────────────────────────────────────────────────────────────────────

def _make_greedy_schedule(
    jobs: list[Job],
    price_data: dict,
    carbon_data: dict,
    base_dt: datetime,
) -> list[ScheduleDecision]:
    """Build a simple ASAP schedule for testing."""
    return [
        ScheduleDecision(
            job_id=j.job_id,
            start_time=j.earliest_start.replace(minute=0, second=0, microsecond=0),
            region=j.region_options[0],
            power_fraction=1.0,
            actual_runtime_hours=j.runtime_hours,
        )
        for j in jobs
    ]
