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

from ..models import Job, ScheduleDecision, ScheduleSegment, OptimizationConfig
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
        elif method == "greedy_migrate":
            # First solve single-segment with greedy, then try to improve each
            # migratable job with a single-split (region migration mid-job).
            result = self._solve_greedy(jobs, price_data, carbon_data, risk_data)
            result = self._apply_migration_optimization(result, jobs, price_data)
        elif method == "greedy_migrate_dp":
            # First solve single-segment with greedy, then run an exact DP per
            # job over (useful_hours_done, current_region, num_migrations) to
            # find the globally-optimal multi-migration sequence. Captures
            # daily/weekly cycle chasing on long jobs that single-migration cannot.
            result = self._solve_greedy(jobs, price_data, carbon_data, risk_data)
            result = self._apply_migration_optimization(
                result, jobs, price_data, mode="dp",
            )
        elif method == "local_search_migrate":
            result = self._solve_local_search(
                jobs, price_data, carbon_data, risk_data, time_limit_seconds
            )
            result = self._apply_migration_optimization(result, jobs, price_data)
        elif method == "local_search_migrate_dp":
            result = self._solve_local_search(
                jobs, price_data, carbon_data, risk_data, time_limit_seconds
            )
            result = self._apply_migration_optimization(
                result, jobs, price_data, mode="dp",
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

    # ------------------------------------------------------------------
    # Mid-job region migration support
    # ------------------------------------------------------------------

    def _apply_migration_optimization(
        self,
        result: "SchedulerResult",
        jobs: list[Job],
        price_data: dict[str, dict[datetime, float]],
        mode: str = "single",
    ) -> "SchedulerResult":
        """Post-process a single-segment schedule by adding region migrations.

        Two modes:
          - "single": heuristic — try every (split_hour, dest_region) and keep
                      the best single-split. Fast; matches greedy_migrate.
          - "dp":     exact optimization — DP over (useful_hours_done,
                      current_region, num_migrations). Captures multi-migration
                      sequences (e.g. chase daily price cycles across the full
                      runtime of a multi-day training job). Used by
                      greedy_migrate_dp / local_search_migrate_dp.

        Both modes use forecast prices for the decision, just like the base
        greedy. The evaluator scores with actual prices via the segment-aware
        accounting in models.ScheduleDecision.all_segments.
        """
        job_by_id = {j.job_id: j for j in jobs}
        improved: list[ScheduleDecision] = []
        migrations_added = 0
        total_extra_migrations = 0

        for decision in result.schedule:
            job = job_by_id.get(decision.job_id)
            if job is None:
                improved.append(decision)
                continue

            if mode == "dp":
                improved_decision = self._try_optimal_migrations(decision, job, price_data)
            else:
                improved_decision = self._try_single_migration(decision, job, price_data)

            extra = improved_decision.migration_count - decision.migration_count
            if extra > 0:
                migrations_added += 1
                total_extra_migrations += extra
            improved.append(improved_decision)

        if migrations_added:
            logger.info(
                f"Migration optimization ({mode}): added migrations to "
                f"{migrations_added} of {len(result.schedule)} jobs "
                f"(total new migrations: {total_extra_migrations})"
            )

        result.schedule = improved
        return result

    def _try_single_migration(
        self,
        decision: ScheduleDecision,
        job: Job,
        price_data: dict[str, dict[datetime, float]],
    ) -> ScheduleDecision:
        """Try every single split + dest-region. Return best (possibly unchanged)."""
        if job.migration_cost_hours is None:
            return decision
        if len(job.region_options) < 2:
            return decision
        if decision.actual_runtime_hours < 2:
            # Need at least 2h to split meaningfully
            return decision

        initial_region = decision.region
        candidate_regions = [r for r in job.region_options if r != initial_region]
        if not candidate_regions:
            return decision

        power_kw = job.power_kw * decision.power_fraction
        runtime = decision.actual_runtime_hours
        migration_cost = job.migration_cost_hours

        original_cost = self._segment_forecast_cost(
            decision.start_time, runtime, initial_region, power_kw, price_data,
        )
        best_cost = original_cost
        best_decision = decision

        for split_h in range(1, int(runtime)):
            useful_after_split = runtime - split_h
            for dest_region in candidate_regions:
                seg1_start = decision.start_time
                seg1_end = decision.start_time + timedelta(hours=split_h)
                # Segment 2 includes migration_cost_hours of warmup at destination
                # at the start of the segment, then useful work. We account for
                # this by making segment 2's duration = migration_cost + useful_after_split.
                seg2_start = seg1_end
                seg2_duration = migration_cost + useful_after_split
                seg2_end = seg2_start + timedelta(hours=seg2_duration)

                # Deadline feasibility: total wallclock must fit
                if seg2_end > job.deadline:
                    continue

                cost1 = self._segment_forecast_cost(
                    seg1_start, split_h, initial_region, power_kw, price_data,
                )
                cost2 = self._segment_forecast_cost(
                    seg2_start, seg2_duration, dest_region, power_kw, price_data,
                )
                total = cost1 + cost2

                if total < best_cost:
                    best_cost = total
                    best_decision = ScheduleDecision(
                        job_id=decision.job_id,
                        start_time=seg1_start,
                        region=initial_region,
                        power_fraction=decision.power_fraction,
                        actual_runtime_hours=runtime + migration_cost,
                        forecast=decision.forecast,
                        segments=[
                            ScheduleSegment(
                                start_time=seg1_start,
                                end_time=seg1_end,
                                region=initial_region,
                                power_fraction=decision.power_fraction,
                            ),
                            ScheduleSegment(
                                start_time=seg2_start,
                                end_time=seg2_end,
                                region=dest_region,
                                power_fraction=decision.power_fraction,
                            ),
                        ],
                    )

        return best_decision

    def _try_optimal_migrations(
        self,
        decision: ScheduleDecision,
        job: Job,
        price_data: dict[str, dict[datetime, float]],
        max_migrations_cap: int = 20,
    ) -> ScheduleDecision:
        """Exact DP for the multi-migration sub-problem given a fixed start_time.

        State:   (u, r, k) — useful_hours_completed, current_region_index, num_migrations
        Wallclock invariant at state (u, r, k): T_0 + (u + k*m) hours, where m
        is migration_cost_hours. This holds for every path to (u, r, k) because
        every useful hour adds 1h wallclock and every migration adds m.

        Transitions from (u, r, k):
          stay     : (u+1, r,  k)    cost = forecast(r, [t, t+1h))
          migrate  : (u+1, r', k+1)  cost = forecast(r', [t, t+m)) + forecast(r', [t+m, t+m+1h))
            (for each r' != r; requires k+1 <= K_max)

        Initial: dp[0][r][0] = 0 for all r (DP picks optimal starting region).
        Terminal: min over (r, k) of dp[H][r][k].

        K_max is bounded by both deadline-feasibility and a configurable cap.
        Reconstruction via parent pointers, walking backward from optimal terminal.
        """
        if job.migration_cost_hours is None:
            return decision
        if len(job.region_options) < 2:
            return decision
        # Need at least 2 useful hours for any migration to make sense
        H = int(decision.actual_runtime_hours)
        if H < 2:
            return decision

        m = float(job.migration_cost_hours)
        T_0 = decision.start_time
        P = job.power_kw * decision.power_fraction
        D = job.deadline
        regions = list(job.region_options)
        R = len(regions)

        # K_max from deadline feasibility:
        #   final wallclock = H + K*m  must  <= (D - T_0)_hours
        deadline_hours = (D - T_0).total_seconds() / 3600.0
        if m > 0:
            k_max_feasible = int(max(0, (deadline_hours - H) // m))
        else:
            k_max_feasible = H  # free migrations — bounded by useful hours
        K_max = min(max_migrations_cap, k_max_feasible)

        if K_max == 0:
            # No migrations feasible
            return decision

        INF = float("inf")
        # dp[u][r][k] and parent[u][r][k]
        dp = [[[INF] * (K_max + 1) for _ in range(R)] for _ in range(H + 1)]
        parent: list[list[list[Optional[tuple]]]] = [
            [[None] * (K_max + 1) for _ in range(R)] for _ in range(H + 1)
        ]

        # Initial: DP picks any starting region at no extra cost
        for r_idx in range(R):
            dp[0][r_idx][0] = 0.0

        # Forward DP
        for u in range(H):
            for r_idx in range(R):
                for k in range(K_max + 1):
                    cur_cost = dp[u][r_idx][k]
                    if cur_cost == INF:
                        continue
                    wall_now = u + k * m  # hours from T_0
                    current_dt = T_0 + timedelta(hours=wall_now)

                    # --- Stay transition ---
                    stay_cost = self._segment_forecast_cost(
                        current_dt, 1.0, regions[r_idx], P, price_data,
                    )
                    cand = cur_cost + stay_cost
                    if cand < dp[u + 1][r_idx][k]:
                        dp[u + 1][r_idx][k] = cand
                        parent[u + 1][r_idx][k] = (u, r_idx, k, "stay")

                    # --- Migrate transition (each other region) ---
                    if k < K_max:
                        for r2_idx in range(R):
                            if r2_idx == r_idx:
                                continue
                            # Warmup at destination for m hours, then 1 useful hour
                            warmup_cost = self._segment_forecast_cost(
                                current_dt, m, regions[r2_idx], P, price_data,
                            )
                            useful_dt = current_dt + timedelta(hours=m)
                            useful_cost = self._segment_forecast_cost(
                                useful_dt, 1.0, regions[r2_idx], P, price_data,
                            )
                            cand = cur_cost + warmup_cost + useful_cost
                            if cand < dp[u + 1][r2_idx][k + 1]:
                                dp[u + 1][r2_idx][k + 1] = cand
                                parent[u + 1][r2_idx][k + 1] = (u, r_idx, k, "migrate")

        # Find best terminal state
        best_cost = INF
        best_r_idx = 0
        best_k = 0
        for r_idx in range(R):
            for k in range(K_max + 1):
                if dp[H][r_idx][k] < best_cost:
                    best_cost = dp[H][r_idx][k]
                    best_r_idx = r_idx
                    best_k = k

        # Compare to original single-segment cost. Only adopt DP solution if
        # it's strictly better (avoids gratuitous segmentation on ties).
        original_cost = self._segment_forecast_cost(
            decision.start_time, float(decision.actual_runtime_hours),
            decision.region, P, price_data,
        )
        if best_cost >= original_cost:
            return decision

        # Reconstruct path: walk back from (H, best_r_idx, best_k) to u=0
        # Collect (next_u, next_r, next_k, action) tuples in reverse order.
        reverse_path: list[tuple[int, int, int, str]] = []
        u, r_idx, k = H, best_r_idx, best_k
        while u > 0:
            p = parent[u][r_idx][k]
            if p is None:
                # Should not happen if dp[H][best_r_idx][best_k] < INF
                return decision
            prev_u, prev_r, prev_k, action = p
            reverse_path.append((u, r_idx, k, action))
            u, r_idx, k = prev_u, prev_r, prev_k

        forward_path = list(reversed(reverse_path))
        initial_r_idx = r_idx  # u==0 region

        # Build segment list by walking forward
        segments: list[ScheduleSegment] = []
        seg_start_wall = 0.0
        seg_region_idx = initial_r_idx
        # Track current state as we walk
        cur_u, cur_r_idx, cur_k = 0, initial_r_idx, 0

        for (next_u, next_r_idx, next_k, action) in forward_path:
            if action == "migrate":
                # Close current segment at the wallclock where migration starts
                wall_at_migration = cur_u + cur_k * m
                segments.append(ScheduleSegment(
                    start_time=T_0 + timedelta(hours=seg_start_wall),
                    end_time=T_0 + timedelta(hours=wall_at_migration),
                    region=regions[seg_region_idx],
                    power_fraction=decision.power_fraction,
                ))
                seg_start_wall = wall_at_migration
                seg_region_idx = next_r_idx
            # Advance state (both stay and migrate end at (next_u, next_r_idx, next_k))
            cur_u, cur_r_idx, cur_k = next_u, next_r_idx, next_k

        # Final segment: from current seg_start_wall to terminal wallclock
        final_wall = H + best_k * m
        segments.append(ScheduleSegment(
            start_time=T_0 + timedelta(hours=seg_start_wall),
            end_time=T_0 + timedelta(hours=final_wall),
            region=regions[seg_region_idx],
            power_fraction=decision.power_fraction,
        ))

        return ScheduleDecision(
            job_id=decision.job_id,
            start_time=T_0,
            region=regions[initial_r_idx],
            power_fraction=decision.power_fraction,
            actual_runtime_hours=H + best_k * m,
            forecast=decision.forecast,
            segments=segments,
        )

    @staticmethod
    def _segment_forecast_cost(
        start: datetime,
        duration_hours: float,
        region: str,
        power_kw: float,
        price_data: dict[str, dict[datetime, float]],
        fallback_price: float = 50.0,
    ) -> float:
        """Sum forecasted energy cost over a [start, start+duration) window."""
        cost = 0.0
        remaining = duration_hours
        current = start.replace(minute=0, second=0, microsecond=0)
        region_prices = price_data.get(region, {})
        while remaining > 0:
            hour_frac = min(1.0, remaining)
            price = region_prices.get(current, fallback_price)
            # price [$/MWh] * power [kW] / 1000 * hours = $
            cost += (price / 1000.0) * power_kw * hour_frac
            remaining -= hour_frac
            current = current + timedelta(hours=1)
        return cost

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
