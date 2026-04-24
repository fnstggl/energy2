"""Constraint definitions for job scheduling optimization.

Constraints ensure schedules are feasible:
1. Time constraints: jobs start within allowed window
2. Deadline constraints: jobs complete by deadline
3. Power constraints: regional power caps respected
4. Runtime constraints: throttling affects runtime
"""

from datetime import datetime, timedelta
from typing import Optional
from dataclasses import dataclass
import logging

from ..models import Job, ScheduleDecision, OptimizationConfig

logger = logging.getLogger(__name__)


@dataclass
class ConstraintViolation:
    """Represents a constraint violation.

    Attributes:
        job_id: The job that violates constraints
        constraint_type: Type of constraint violated
        message: Description of violation
        severity: How severe the violation is (0-1)
    """
    job_id: str
    constraint_type: str
    message: str
    severity: float = 1.0


class ConstraintBuilder:
    """Builds and checks scheduling constraints.

    Constraints:
    1. earliest_start: job cannot start before earliest_start
    2. deadline: job must complete by deadline
    3. region_valid: job must run in allowed region
    4. power_valid: power_fraction in valid range
    5. power_cap: regional power cap not exceeded
    """

    def __init__(
        self,
        config: Optional[OptimizationConfig] = None,
    ):
        """Initialize constraint builder.

        Args:
            config: Optimization configuration
        """
        self.config = config or OptimizationConfig()

    def check_job_constraints(
        self,
        job: Job,
        decision: ScheduleDecision,
    ) -> list[ConstraintViolation]:
        """Check all constraints for a single job decision.

        Args:
            job: The job specification
            decision: The scheduling decision

        Returns:
            List of constraint violations (empty if valid)
        """
        violations = []

        # 1. Earliest start constraint
        if decision.start_time < job.earliest_start:
            delta = (job.earliest_start - decision.start_time).total_seconds() / 3600
            violations.append(ConstraintViolation(
                job_id=job.job_id,
                constraint_type="earliest_start",
                message=f"Starts {delta:.2f} hours before earliest allowed",
                severity=min(1.0, delta / 24),
            ))

        # 2. Deadline constraint
        end_time = decision.end_time
        if end_time > job.deadline:
            delta = (end_time - job.deadline).total_seconds() / 3600
            violations.append(ConstraintViolation(
                job_id=job.job_id,
                constraint_type="deadline",
                message=f"Finishes {delta:.2f} hours after deadline",
                severity=min(1.0, delta / 24),
            ))

        # 3. Region validity
        if decision.region not in job.region_options:
            violations.append(ConstraintViolation(
                job_id=job.job_id,
                constraint_type="region_valid",
                message=f"Region {decision.region} not in allowed regions {job.region_options}",
                severity=1.0,
            ))

        # 4. Power fraction validity
        if decision.power_fraction < self.config.min_power_fraction:
            violations.append(ConstraintViolation(
                job_id=job.job_id,
                constraint_type="power_min",
                message=f"Power fraction {decision.power_fraction} below minimum {self.config.min_power_fraction}",
                severity=0.5,
            ))
        if decision.power_fraction > self.config.max_power_fraction:
            violations.append(ConstraintViolation(
                job_id=job.job_id,
                constraint_type="power_max",
                message=f"Power fraction {decision.power_fraction} above maximum {self.config.max_power_fraction}",
                severity=0.5,
            ))

        return violations

    def check_schedule_constraints(
        self,
        jobs: list[Job],
        schedule: list[ScheduleDecision],
    ) -> list[ConstraintViolation]:
        """Check all constraints for a complete schedule.

        Args:
            jobs: List of jobs
            schedule: List of scheduling decisions

        Returns:
            List of all constraint violations
        """
        job_by_id = {j.job_id: j for j in jobs}
        all_violations = []

        for decision in schedule:
            job = job_by_id.get(decision.job_id)
            if not job:
                all_violations.append(ConstraintViolation(
                    job_id=decision.job_id,
                    constraint_type="job_missing",
                    message=f"Job {decision.job_id} not found in job list",
                    severity=1.0,
                ))
                continue
            violations = self.check_job_constraints(job, decision)
            all_violations.extend(violations)

        # Check regional power caps
        power_violations = self._check_power_caps(jobs, schedule)
        all_violations.extend(power_violations)

        return all_violations

    def _check_power_caps(
        self,
        jobs: list[Job],
        schedule: list[ScheduleDecision],
    ) -> list[ConstraintViolation]:
        """Check that regional power caps are respected.

        For each hour, sum power usage per region and check against caps.

        Args:
            jobs: List of jobs
            schedule: List of scheduling decisions

        Returns:
            List of power cap violations
        """
        job_by_id = {j.job_id: j for j in jobs}
        violations = []

        # Find time range
        if not schedule:
            return violations

        start_times = [d.start_time for d in schedule]
        end_times = [d.end_time for d in schedule]
        earliest = min(start_times)
        latest = max(end_times)

        # Check each hour
        current = earliest.replace(minute=0, second=0, microsecond=0)
        while current <= latest:
            # Sum power by region for this hour
            power_by_region: dict[str, float] = {}

            for decision in schedule:
                # Is this job running during this hour?
                if decision.start_time <= current < decision.end_time:
                    job = job_by_id.get(decision.job_id)
                    if job:
                        power = job.power_kw * decision.power_fraction
                        power_by_region[decision.region] = power_by_region.get(decision.region, 0) + power

            # Check against caps
            for region, total_power in power_by_region.items():
                cap = self.config.region_power_caps.get(region, float('inf'))
                if total_power > cap:
                    violations.append(ConstraintViolation(
                        job_id="multiple",
                        constraint_type="power_cap",
                        message=f"Region {region} at {current}: {total_power:.0f} kW exceeds cap {cap:.0f} kW",
                        severity=min(1.0, (total_power - cap) / cap),
                    ))

            current += timedelta(hours=1)

        return violations

    def is_feasible(
        self,
        jobs: list[Job],
        schedule: list[ScheduleDecision],
    ) -> bool:
        """Check if a schedule is feasible (no violations).

        Args:
            jobs: List of jobs
            schedule: List of scheduling decisions

        Returns:
            True if schedule has no violations
        """
        violations = self.check_schedule_constraints(jobs, schedule)
        return len(violations) == 0

    def get_feasible_start_range(
        self,
        job: Job,
        power_fraction: float = 1.0,
    ) -> tuple[datetime, datetime]:
        """Get the valid start time range for a job.

        Args:
            job: The job
            power_fraction: Power level (affects runtime)

        Returns:
            (earliest_start, latest_start) tuple
        """
        runtime = job.adjusted_runtime(power_fraction)
        latest_start = job.deadline - timedelta(hours=runtime)
        return (job.earliest_start, latest_start)

    def get_feasible_power_range(
        self,
        job: Job,
        start_time: datetime,
    ) -> tuple[float, float]:
        """Get the valid power fraction range for a job starting at a given time.

        Power affects runtime, so there's a minimum power to meet deadline.

        Args:
            job: The job
            start_time: Proposed start time

        Returns:
            (min_power, max_power) tuple
        """
        # Maximum power is always the configured max
        max_power = self.config.max_power_fraction

        # Minimum power is constrained by deadline
        # runtime = base_runtime / power_fraction
        # We need: start_time + runtime <= deadline
        # So: runtime <= deadline - start_time
        # And: base_runtime / power <= available_time
        # Thus: power >= base_runtime / available_time

        available_hours = (job.deadline - start_time).total_seconds() / 3600
        if available_hours <= 0:
            return (max_power, max_power)  # No slack, must run at max

        min_power_for_deadline = job.runtime_hours / available_hours

        # Apply configured minimum
        min_power = max(self.config.min_power_fraction, min_power_for_deadline)

        return (min(min_power, max_power), max_power)

    def would_violate_power_cap(
        self,
        job: Job,
        decision: ScheduleDecision,
        existing_schedule: list[ScheduleDecision],
        all_jobs: list[Job],
    ) -> bool:
        """Return True if adding *decision* would breach any regional power cap.

        Called inside the solver loop so infeasible candidates are skipped
        before the objective is even computed.

        Args:
            job:               The job being placed.
            decision:          The candidate ScheduleDecision to test.
            existing_schedule: Decisions already committed to the schedule.
            all_jobs:          All jobs (needed for power_kw lookup).
        """
        cap = self.config.region_power_caps.get(decision.region)
        if cap is None:
            return False  # no cap configured for this region

        job_by_id = {j.job_id: j for j in all_jobs}
        job_power = job.power_kw * decision.power_fraction

        # Check every hour the candidate job would be running
        current = decision.start_time.replace(minute=0, second=0, microsecond=0)
        end = decision.end_time
        while current < end:
            region_load = job_power
            for existing in existing_schedule:
                if existing.region != decision.region:
                    continue
                if existing.start_time <= current < existing.end_time:
                    existing_job = job_by_id.get(existing.job_id)
                    if existing_job:
                        region_load += existing_job.power_kw * existing.power_fraction
            if region_load > cap:
                return True
            current = current.replace(minute=0, second=0, microsecond=0)
            from datetime import timedelta
            current += timedelta(hours=1)

        return False

    def summarize_violations(
        self,
        violations: list[ConstraintViolation],
    ) -> dict:
        """Summarize constraint violations.

        Args:
            violations: List of violations

        Returns:
            Summary statistics
        """
        if not violations:
            return {
                "total_violations": 0,
                "by_type": {},
                "affected_jobs": [],
                "avg_severity": 0.0,
            }

        by_type: dict[str, int] = {}
        affected_jobs = set()
        total_severity = 0.0

        for v in violations:
            by_type[v.constraint_type] = by_type.get(v.constraint_type, 0) + 1
            affected_jobs.add(v.job_id)
            total_severity += v.severity

        return {
            "total_violations": len(violations),
            "by_type": by_type,
            "affected_jobs": list(affected_jobs),
            "avg_severity": round(total_severity / len(violations), 3),
        }
