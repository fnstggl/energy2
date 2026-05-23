"""Tests for customer workload trace CSV ingestion.

The ``load_from_customer_csv`` and ``load_from_file`` methods implement the
simplified customer-facing CSV schema described in docs/PILOT_READINESS_AUDIT.md.
"""

import io
import textwrap
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd
import pytest

from aurelius.ingestion.job_logs import JobLogIngester
from aurelius.models import Job


UTC = timezone.utc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def write_csv(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "trace.csv"
    p.write_text(textwrap.dedent(content).strip())
    return p


MINIMAL_CSV = """\
    job_id,workload_type,submit_time,duration_hours
    j1,training,2026-01-15T00:00:00Z,48
    j2,llm_batch_inference,2026-01-15T06:00:00Z,4
    j3,realtime_inference,2026-01-15T08:00:00Z,0.5
"""

FULL_CSV = (
    "job_id,workload_type,submit_time,duration_hours,gpu_count,deadline,"
    "max_delay_hours,allowed_regions,forbidden_regions,interruptible,"
    "checkpointable,data_transfer_gb,sla_class,sla_penalty_per_hour,gpu_type,"
    "power_kw,migration_cost_hours,pue\n"
    "j1,training,2026-01-15T00:00:00Z,48,16,2026-01-20T00:00:00Z,72,"
    "us-west|us-east,,1,1,10.0,guaranteed,5.0,a100,6.4,0.75,1.2\n"
)


# ---------------------------------------------------------------------------
# TestLoadFromCustomerCSVMinimal
# ---------------------------------------------------------------------------

class TestLoadFromCustomerCSVMinimal:
    def test_loads_jobs(self, tmp_path):
        p = write_csv(tmp_path, MINIMAL_CSV)
        jobs = JobLogIngester().load_from_customer_csv(p)
        assert len(jobs) == 3

    def test_job_id_preserved(self, tmp_path):
        p = write_csv(tmp_path, MINIMAL_CSV)
        jobs = JobLogIngester().load_from_customer_csv(p)
        assert jobs[0].job_id == "j1"

    def test_workload_type_preserved(self, tmp_path):
        p = write_csv(tmp_path, MINIMAL_CSV)
        jobs = JobLogIngester().load_from_customer_csv(p)
        assert jobs[0].workload_type == "training"
        assert jobs[1].workload_type == "llm_batch_inference"
        assert jobs[2].workload_type == "realtime_inference"

    def test_duration_set(self, tmp_path):
        p = write_csv(tmp_path, MINIMAL_CSV)
        jobs = JobLogIngester().load_from_customer_csv(p)
        assert jobs[0].runtime_hours == 48.0
        assert jobs[2].runtime_hours == 0.5

    def test_submit_time_utc(self, tmp_path):
        p = write_csv(tmp_path, MINIMAL_CSV)
        jobs = JobLogIngester().load_from_customer_csv(p)
        assert jobs[0].submit_time.tzinfo is not None
        assert jobs[0].submit_time == datetime(2026, 1, 15, 0, 0, 0, tzinfo=UTC)

    def test_deadline_derived_from_max_delay(self, tmp_path):
        p = write_csv(tmp_path, MINIMAL_CSV)
        jobs = JobLogIngester().load_from_customer_csv(p)
        # training default: max_delay_hours=48.0; deadline = submit + 48h + 48h
        expected = datetime(2026, 1, 15, tzinfo=UTC) + timedelta(hours=48 + 48)
        assert jobs[0].deadline == expected

    def test_realtime_inference_max_delay_zero(self, tmp_path):
        p = write_csv(tmp_path, MINIMAL_CSV)
        jobs = JobLogIngester().load_from_customer_csv(p)
        rt_job = next(j for j in jobs if j.workload_type == "realtime_inference")
        assert rt_job.max_delay_hours == 0.0

    def test_default_gpu_count_from_workload_type(self, tmp_path):
        p = write_csv(tmp_path, MINIMAL_CSV)
        jobs = JobLogIngester().load_from_customer_csv(p)
        train = next(j for j in jobs if j.workload_type == "training")
        assert train.gpu_count == 8  # training default

    def test_power_kw_estimated_from_gpu_count(self, tmp_path):
        p = write_csv(tmp_path, MINIMAL_CSV)
        jobs = JobLogIngester().load_from_customer_csv(p)
        train = next(j for j in jobs if j.workload_type == "training")
        # 8 GPUs * 0.4 kW/GPU = 3.2 kW
        assert abs(train.power_kw - 3.2) < 0.01

    def test_training_interruptible_true(self, tmp_path):
        p = write_csv(tmp_path, MINIMAL_CSV)
        jobs = JobLogIngester().load_from_customer_csv(p)
        train = next(j for j in jobs if j.workload_type == "training")
        assert train.interruptible is True

    def test_realtime_inference_not_interruptible(self, tmp_path):
        p = write_csv(tmp_path, MINIMAL_CSV)
        jobs = JobLogIngester().load_from_customer_csv(p)
        rt = next(j for j in jobs if j.workload_type == "realtime_inference")
        assert rt.interruptible is False

    def test_default_regions_applied(self, tmp_path):
        p = write_csv(tmp_path, MINIMAL_CSV)
        jobs = JobLogIngester().load_from_customer_csv(p)
        assert "us-west" in jobs[0].region_options
        assert "us-east" in jobs[0].region_options

    def test_custom_default_regions(self, tmp_path):
        p = write_csv(tmp_path, MINIMAL_CSV)
        jobs = JobLogIngester().load_from_customer_csv(p, default_regions=["us-west"])
        assert jobs[0].region_options == ["us-west"]


# ---------------------------------------------------------------------------
# TestLoadFromCustomerCSVFull
# ---------------------------------------------------------------------------

class TestLoadFromCustomerCSVFull:
    def _load(self, tmp_path):
        p = tmp_path / "full_trace.csv"
        p.write_text(FULL_CSV)
        return JobLogIngester().load_from_customer_csv(p)[0]

    def test_gpu_count_from_csv(self, tmp_path):
        job = self._load(tmp_path)
        assert job.gpu_count == 16

    def test_power_kw_from_csv(self, tmp_path):
        job = self._load(tmp_path)
        assert abs(job.power_kw - 6.4) < 0.01

    def test_deadline_from_csv(self, tmp_path):
        job = self._load(tmp_path)
        assert job.deadline == datetime(2026, 1, 20, 0, 0, 0, tzinfo=UTC)

    def test_max_delay_from_csv(self, tmp_path):
        job = self._load(tmp_path)
        assert job.max_delay_hours == 72.0

    def test_allowed_regions_from_csv(self, tmp_path):
        job = self._load(tmp_path)
        assert "us-west" in job.allowed_regions
        assert "us-east" in job.allowed_regions

    def test_interruptible_from_csv(self, tmp_path):
        job = self._load(tmp_path)
        assert job.interruptible is True

    def test_sla_class_from_csv(self, tmp_path):
        job = self._load(tmp_path)
        assert job.sla_class == "guaranteed"

    def test_sla_penalty_from_csv(self, tmp_path):
        job = self._load(tmp_path)
        assert job.sla_penalty_per_hour == 5.0

    def test_gpu_type_from_csv(self, tmp_path):
        job = self._load(tmp_path)
        assert job.gpu_type == "a100"

    def test_migration_cost_from_csv(self, tmp_path):
        job = self._load(tmp_path)
        assert abs(job.migration_cost_hours - 0.75) < 0.001

    def test_pue_from_csv(self, tmp_path):
        job = self._load(tmp_path)
        assert abs(job.pue - 1.2) < 0.001

    def test_data_transfer_gb_from_csv(self, tmp_path):
        job = self._load(tmp_path)
        assert abs(job.data_transfer_gb - 10.0) < 0.001


# ---------------------------------------------------------------------------
# TestLoadFromCustomerCSVValidation
# ---------------------------------------------------------------------------

class TestLoadFromCustomerCSVValidation:
    def test_missing_required_column_raises(self, tmp_path):
        csv = "job_id,workload_type,submit_time\nj1,training,2026-01-01T00:00:00Z"
        p = write_csv(tmp_path, csv)
        with pytest.raises(ValueError, match="missing required columns"):
            JobLogIngester().load_from_customer_csv(p)

    def test_unknown_workload_type_raises(self, tmp_path):
        csv = "job_id,workload_type,submit_time,duration_hours\nj1,gpu_mining,2026-01-01T00:00:00Z,24"
        p = write_csv(tmp_path, csv)
        with pytest.raises(ValueError, match="Unknown workload_type"):
            JobLogIngester().load_from_customer_csv(p)

    def test_all_seven_workload_types_accepted(self, tmp_path):
        types = [
            "training", "fine_tuning", "llm_batch_inference",
            "data_processing", "scheduled_batch", "realtime_inference",
            "background_maintenance",
        ]
        rows = "\n".join(
            f"j{i},{wt},2026-01-0{i+1}T00:00:00Z,{(i+1)*2}"
            for i, wt in enumerate(types)
        )
        csv = "job_id,workload_type,submit_time,duration_hours\n" + rows
        p = write_csv(tmp_path, csv)
        jobs = JobLogIngester().load_from_customer_csv(p)
        assert len(jobs) == 7
        loaded_types = {j.workload_type for j in jobs}
        assert loaded_types == set(types)


# ---------------------------------------------------------------------------
# TestLoadFromFile (auto-detect)
# ---------------------------------------------------------------------------

class TestLoadFromFile:
    def test_detects_customer_csv_by_columns(self, tmp_path):
        p = write_csv(tmp_path, MINIMAL_CSV)
        jobs = JobLogIngester().load_from_file(p)
        assert len(jobs) == 3
        assert all(isinstance(j, Job) for j in jobs)

    def test_detects_json_by_extension(self, tmp_path):
        import json
        data = [
            {
                "job_id": "j1",
                "submit_time": "2026-01-15T00:00:00+00:00",
                "runtime_hours": 4.0,
                "deadline": "2026-01-16T00:00:00+00:00",
                "power_kw": 1.6,
                "earliest_start": "2026-01-15T00:00:00+00:00",
                "region_options": ["us-west"],
                "workload_type": "training",
            }
        ]
        p = tmp_path / "jobs.json"
        p.write_text(json.dumps(data))
        jobs = JobLogIngester().load_from_file(p)
        assert len(jobs) == 1
        assert jobs[0].job_id == "j1"

    def test_csv_result_compatible_with_job_model(self, tmp_path):
        p = write_csv(tmp_path, MINIMAL_CSV)
        jobs = JobLogIngester().load_from_file(p)
        for j in jobs:
            assert j.job_id
            assert j.submit_time.tzinfo is not None
            assert j.deadline > j.submit_time
            assert j.runtime_hours > 0
            assert j.power_kw > 0
            assert len(j.region_options) > 0


# ---------------------------------------------------------------------------
# TestDefaultsPerWorkloadType
# ---------------------------------------------------------------------------

class TestDefaultsPerWorkloadType:
    def _job_for(self, wtype: str, tmp_path: Path) -> Job:
        csv = f"job_id,workload_type,submit_time,duration_hours\nj1,{wtype},2026-01-15T00:00:00Z,24"
        p = write_csv(tmp_path, csv)
        return JobLogIngester().load_from_customer_csv(p)[0]

    def test_training_defaults(self, tmp_path):
        j = self._job_for("training", tmp_path)
        assert j.gpu_count == 8
        assert j.interruptible is True
        assert j.checkpointable is True
        assert j.max_delay_hours == 48.0
        assert j.migration_cost_hours == 0.5

    def test_realtime_inference_defaults(self, tmp_path):
        j = self._job_for("realtime_inference", tmp_path)
        assert j.gpu_count == 2
        assert j.interruptible is False
        assert j.max_delay_hours == 0.0
        assert j.migration_cost_hours is None

    def test_background_maintenance_defaults(self, tmp_path):
        j = self._job_for("background_maintenance", tmp_path)
        assert j.max_delay_hours == 168.0
        assert j.interruptible is True

    def test_llm_batch_inference_not_interruptible(self, tmp_path):
        j = self._job_for("llm_batch_inference", tmp_path)
        assert j.interruptible is False
        assert j.checkpointable is True

    def test_fine_tuning_defaults(self, tmp_path):
        j = self._job_for("fine_tuning", tmp_path)
        assert j.gpu_count == 4
        assert j.max_delay_hours == 24.0
        assert j.migration_cost_hours == 0.25
