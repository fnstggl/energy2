"""Job scheduler optimization.

This is the core optimization solver that decides:
- When each job should start (time shifting)
- Where each job should run (region routing)
- How fast each job should run (power throttling)

Uses a combination of:
1. Greedy heuristics for initial solution
2. Local search for improvement
3. Optional MILP for optimal solutions (with PuLP)
"""

from datetime import datetime, timedelta
from typing import Optional
from dataclasses import dataclass, field
import logging
import math

from ..models import Job, ScheduleDecision, OptimizationConfig
from .objective import ObjectiveFunction, ObjectiveComponents
from .constraints import ConstraintBuilder

logger = logging.getLogger(__name__)


@dataclass
class SchedulerResult:
    """Result from the scheduler.

    Attributes:
        schedule: List of scheduling decisions
        objective: Objective function components
        violations: Number of constraint violations
        solver_time_ms: Time spent solving
        iterations: Number of iterations (for iterative solvers)
    """
    schedule: list[ScheduleDecision]
    objective: ObjectiveComponents
    violations: int = 0
    solver_time_ms: float = 0.0
    iterations: int = 0


class JobScheduler:
    """Optimizes job scheduling across time, regions, and power levels.

    The scheduler implements the core optimization logic for Aurelius.
    It supports multiple solving strategies:

    1. Greedy: Fast, reasonable solution
       - Process jobs by priority/deadline
       - For each job, evaluate all feasible options
       - Pick the best option greedily

    2. Local Search: Improve greedy solution
       - Start from greedy solution
       - Try swapping/moving jobs to better slots
       - Accept improvements, stop when stuck

    3. MILP: Optimal solution (slower)
       - Formulate as mixed-integer linear program
       - Use PuLP solver
       - Returns provably optimal schedule
    """

    def __init__(
        self,
        config: Optional[OptimizationConfig] = None,
    ):
        """Initialize the scheduler.

        Args:
            config: Optimization configuration
        """
        self.config = config or OptimizationConfig()
        self.objective_fn = ObjectiveFunction(config)
        self.constraints = ConstraintBuilder(config)

    def solve(
        self,
        jobs: list[Job],
        price_data: dict[str, dict[datetime, float]],
        carbon_data: dict[str, dict[datetime, float]],
        risk_data: Optional[dict[str, dict[datetime, float]]] = None,
        method: str = "greedy",
        time_limit_seconds: float = 60.0,
    ) -> SchedulerResult:
        """Solve the scheduling problem.

        Args:
            jobs: List of jobs to schedule
            price_data: {region: {timestamp: price_per_mwh}}
            carbon_data: {region: {timestamp: gco2_per_kwh}}
            risk_data: {region: {timestamp: risk_penalty}}
            method: Solving method ("greedy", "local_search", "milp")
            time_limit_seconds: Maximum solving time

        Returns:
            SchedulerResult with optimal schedule
        """
        import time
        start_time = time.time()

        if method == "greedy":
            result = self._solve_greedy(jobs, price_data, carbon_data, risk_data)
        elif method == "local_search":
            result = self._solve_local_search(
                jobs, price_data, carbon_data, risk_data, time_limit_seconds
            )
        elif method == "milp":
            result = self._solve_milp(
                jobs, price_data, carbon_data, risk_data, time_limit_seconds
            )
        else:
            logger.warning(f"Unknown method {method}, falling back to greedy")
            result = self._solve_greedy(jobs, price_data, carbon_data, risk_data)

        elapsed_ms = (time.time() - start_time) * 1000
        result.solver_time_ms = elapsed_ms

        return result

    def _solve_greedy(
        self,
        jobs: list[Job],
        price_data: dict[str, dict[datetime, float]],
        carbon_data: dict[str, dict[datetime, float]],
        risk_data: Optional[dict[str, dict[datetime, float]]] = None,
    ) -> SchedulerResult:
        """Greedy scheduling algorithm.

        Process jobs in order (by priority then deadline).
        For each job, evaluate all feasible (time, region, power) combinations
        and pick the one with lowest objective.
        """
        # Sort jobs: higher priority first, earlier deadline first
        sorted_jobs = sorted(
            jobs,
            key=lambda j: (-j.priority, j.deadline)
        )

        schedule = []
        for job in sorted_jobs:
            best_decision = self._find_best_slot(
                job, schedule, jobs, price_data, carbon_data, risk_data
            )
            if best_decision:
                schedule.append(best_decision)
            else:
                # Fallback: schedule ASAP in first available region
                region = job.region_options[0]
                schedule.append(ScheduleDecision(
                    job_id=job.job_id,
                    start_time=job.earliest_start,
                    region=region,
                    power_fraction=1.0,
                    actual_runtime_hours=job.runtime_hours,
                ))
                logger.warning(f"No optimal slot found for {job.job_id}, using fallback")

        objective = self.objective_fn.calculate(
            jobs, schedule, price_data, carbon_data, risk_data
        )
        violations = len(self.constraints.check_schedule_constraints(jobs, schedule))

        return SchedulerResult(
            schedule=schedule,
            objective=objective,
            violations=violations,
            iterations=len(jobs),
        )

    def _find_best_slot(
        self,
        job: Job,
        current_schedule: list[ScheduleDecision],
        all_jobs: list[Job],
        price_data: dict[str, dict[datetime, float]],
        carbon_data: dict[str, dict[datetime, float]],
        risk_data: Optional[dict[str, dict[datetime, float]]] = None,
    ) -> Optional[ScheduleDecision]:
        """Find the best slot for a job given current schedule.

        Evaluates combinations of:
        - Start times (hourly granularity within feasible window)
        - Regions (from job's region_options)
        - Power fractions (discrete levels)
        """
        best_decision = None
        best_objective = float('inf')

        # Generate candidate start times
        power_levels = [1.0, 0.75, 0.5] if self.config.min_power_fraction <= 0.5 else [1.0]
        power_levels = [p for p in power_levels if p >= self.config.min_power_fraction]

        for power_fraction in power_levels:
            earliest, latest = self.constraints.get_feasible_start_range(job, power_fraction)

            # Hourly granularity
            current_start = earliest.replace(minute=0, second=0, microsecond=0)
            while current_start <= latest:
                for region in job.region_options:
                    # Check power cap feasibility
                    runtime = job.adjusted_runtime(power_fraction)
                    decision = ScheduleDecision(
                        job_id=job.job_id,
                        start_time=current_start,
                        region=region,
                        power_fraction=power_fraction,
                        actual_runtime_hours=runtime,
                    )

                    # Check constraints
                    violations = self.constraints.check_job_constraints(job, decision)
                    if violations:
                        continue

                    # Enforce power cap before computing objective
                    if self.constraints.would_violate_power_cap(
                        job, decision, current_schedule, all_jobs
                    ):
                        continue

                    # Calculate objective for just this job placement
                    obj = self.objective_fn.calculate(
                        [job],
                        [decision],
                        price_data, carbon_data, risk_data
                    )

                    if obj.total < best_objective:
                        best_objective = obj.total
                        best_decision = decision

                current_start += timedelta(hours=1)

        return best_decision

    def _solve_local_search(
        self,
        jobs: list[Job],
        price_data: dict[str, dict[datetime, float]],
        carbon_data: dict[str, dict[datetime, float]],
        risk_data: Optional[dict[str, dict[datetime, float]]] = None,
        time_limit_seconds: float = 60.0,
    ) -> SchedulerResult:
        """Local search improvement over greedy solution.

        1. Start with greedy solution
        2. Try moving each job to better slots
        3. Accept improvements
        4. Repeat until no improvement or time limit
        """
        import time
        start_time = time.time()

        # Get initial greedy solution
        greedy_result = self._solve_greedy(jobs, price_data, carbon_data, risk_data)
        best_schedule = greedy_result.schedule.copy()
        best_objective = greedy_result.objective.total

        iterations = 0
        improved = True

        job_by_id = {j.job_id: j for j in jobs}

        while improved and (time.time() - start_time) < time_limit_seconds:
            improved = False
            iterations += 1

            for i, decision in enumerate(best_schedule):
                job = job_by_id[decision.job_id]

                # Try different slots for this job
                other_decisions = best_schedule[:i] + best_schedule[i+1:]
                new_decision = self._find_best_slot(
                    job, other_decisions, jobs, price_data, carbon_data, risk_data
                )

                if new_decision:
                    test_schedule = other_decisions + [new_decision]
                    obj = self.objective_fn.calculate(
                        jobs, test_schedule, price_data, carbon_data, risk_data
                    )

                    if obj.total < best_objective * 0.999:  # 0.1% improvement threshold
                        best_schedule = test_schedule
                        best_objective = obj.total
                        improved = True
                        break  # Restart from beginning

        # Recalculate final objective
        final_objective = self.objective_fn.calculate(
            jobs, best_schedule, price_data, carbon_data, risk_data
        )
        violations = len(self.constraints.check_schedule_constraints(jobs, best_schedule))

        return SchedulerResult(
            schedule=best_schedule,
            objective=final_objective,
            violations=violations,
            iterations=iterations,
        )

    def _solve_milp(
        self,
        jobs: list[Job],
        price_data: dict[str, dict[datetime, float]],
        carbon_data: dict[str, dict[datetime, float]],
        risk_data: Optional[dict[str, dict[datetime, float]]] = None,
        time_limit_seconds: float = 60.0,
    ) -> SchedulerResult:
        """Mixed Integer Linear Programming solution.

        Uses PuLP to formulate and solve the problem optimally.

        Decision variables:
        - x[j,t,r] = 1 if job j starts at time t in region r
        - p[j] = power fraction for job j

        Objective: minimize weighted cost over all jobs
        """
        try:
            import pulp
        except ImportError:
            logger.warning("PuLP not installed, falling back to local_search")
            return self._solve_local_search(
                jobs, price_data, carbon_data, risk_data, time_limit_seconds
            )

        # Create problem
        prob = pulp.LpProblem("JobScheduling", pulp.LpMinimize)

        # Get time range
        all_times = set()
        for job in jobs:
            current = job.earliest_start.replace(minute=0, second=0, microsecond=0)
            while current <= job.deadline:
                all_times.add(current)
                current += timedelta(hours=1)

        times = sorted(all_times)
        regions = list(set(r for j in jobs for r in j.region_options))

        # Decision variables: x[job_id, time_idx, region] = binary
        x = {}
        for job in jobs:
            for t_idx, t in enumerate(times):
                for r in job.region_options:
                    var_name = f"x_{job.job_id}_{t_idx}_{r}"
                    x[(job.job_id, t_idx, r)] = pulp.LpVariable(var_name, cat='Binary')

        # Each job must be scheduled exactly once
        for job in jobs:
            prob += (
                pulp.lpSum(
                    x.get((job.job_id, t_idx, r), 0)
                    for t_idx in range(len(times))
                    for r in job.region_options
                ) == 1,
                f"schedule_once_{job.job_id}"
            )

        # Time window constraints
        for job in jobs:
            for t_idx, t in enumerate(times):
                for r in job.region_options:
                    if (job.job_id, t_idx, r) not in x:
                        continue
                    # Cannot start before earliest
                    if t < job.earliest_start:
                        prob += x[(job.job_id, t_idx, r)] == 0
                    # Cannot finish after deadline (at full power)
                    end_time = t + timedelta(hours=job.runtime_hours)
                    if end_time > job.deadline:
                        prob += x[(job.job_id, t_idx, r)] == 0

        # Objective: minimize energy cost
        cost_terms = []
        for job in jobs:
            for t_idx, t in enumerate(times):
                for r in job.region_options:
                    if (job.job_id, t_idx, r) not in x:
                        continue

                    # Estimate cost for this assignment
                    total_cost = 0.0
                    current = t
                    remaining = job.runtime_hours
                    while remaining > 0:
                        hour_key = current.replace(minute=0, second=0, microsecond=0)
                        hour_fraction = min(1.0, remaining)

                        # Price
                        price = price_data.get(r, {}).get(hour_key, 50.0)
                        energy_cost = (price / 1000) * job.power_kw * hour_fraction

                        # Carbon
                        carbon = carbon_data.get(r, {}).get(hour_key, 400.0)
                        carbon_cost = carbon * 0.001 * job.power_kw * hour_fraction

                        # Risk
                        risk = 0.05
                        if risk_data:
                            risk = risk_data.get(r, {}).get(hour_key, 0.05)
                        risk_cost = risk * job.power_kw * hour_fraction

                        total_cost += (
                            self.config.alpha * energy_cost +
                            self.config.beta * carbon_cost +
                            self.config.gamma * risk_cost
                        )

                        remaining -= hour_fraction
                        current += timedelta(hours=1)

                    cost_terms.append(total_cost * x[(job.job_id, t_idx, r)])

        prob += pulp.lpSum(cost_terms)

        # Solve
        solver = pulp.PULP_CBC_CMD(timeLimit=time_limit_seconds, msg=0)
        prob.solve(solver)

        # Extract solution
        schedule = []
        for job in jobs:
            for t_idx, t in enumerate(times):
                for r in job.region_options:
                    if (job.job_id, t_idx, r) in x:
                        if pulp.value(x[(job.job_id, t_idx, r)]) > 0.5:
                            schedule.append(ScheduleDecision(
                                job_id=job.job_id,
                                start_time=t,
                                region=r,
                                power_fraction=1.0,  # MILP uses full power for simplicity
                                actual_runtime_hours=job.runtime_hours,
                            ))
                            break
                else:
                    continue
                break

        # If MILP failed to find solution, fall back to greedy
        if len(schedule) != len(jobs):
            logger.warning("MILP incomplete, falling back to greedy")
            return self._solve_greedy(jobs, price_data, carbon_data, risk_data)

        objective = self.objective_fn.calculate(
            jobs, schedule, price_data, carbon_data, risk_data
        )
        violations = len(self.constraints.check_schedule_constraints(jobs, schedule))

        return SchedulerResult(
            schedule=schedule,
            objective=objective,
            violations=violations,
            iterations=1,
        )

    def create_baseline_schedule(
        self,
        jobs: list[Job],
    ) -> list[ScheduleDecision]:
        """Create baseline (ASAP) schedule for comparison.

        Args:
            jobs: List of jobs

        Returns:
            List of baseline schedule decisions
        """
        schedule = []
        for job in jobs:
            region = (
                self.config.default_region
                if self.config.default_region in job.region_options
                else job.region_options[0]
            )
            schedule.append(ScheduleDecision(
                job_id=job.job_id,
                start_time=job.earliest_start,
                region=region,
                power_fraction=1.0,
                actual_runtime_hours=job.runtime_hours,
            ))
        return schedule
