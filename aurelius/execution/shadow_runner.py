"""Shadow runner for production-like evaluation of Aurelius decisions.

ShadowRunner replays scheduling decisions against real (realized) price and
carbon data to compute what Aurelius would have spent vs. what a baseline
would have spent — without modifying any live infrastructure.

All computations are read-only with respect to the scheduling system.
Results are logged and returned as a ShadowResult.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from ..models import Job, OptimizationConfig, ScheduleDecision
from ..optimization.objective import ObjectiveFunction

logger = logging.getLogger(__name__)


@dataclass
class ShadowDecisionRecord:
    """Per-job record of shadow evaluation.

    Captures what Aurelius decided, what the baseline would have decided,
    and what costs actually materialized at realized prices/carbon.
    """
    job_id: str
    workload_type: str
    region: str
    start_time: datetime
    end_time: datetime
    runtime_hours: float
    power_kw: float
    pue: float

    # Aurelius realized costs (using real prices/carbon)
    realized_energy_cost: float
    realized_carbon_kg: float
    realized_sla_penalty: float
    realized_data_transfer_cost: float
    realized_total_cost: float

    # Baseline realized costs (same pricing, different decision)
    baseline_energy_cost: float
    baseline_carbon_kg: float
    baseline_sla_penalty: float
    baseline_data_transfer_cost: float
    baseline_total_cost: float

    # Computed savings
    cost_savings: float       # baseline_total - realized_total (positive = Aurelius won)
    carbon_savings_kg: float  # baseline_carbon - realized_carbon

    # Constraint violations
    constraint_violations: list[str]

    # Forecast snapshot (if available at decision time)
    forecast_snapshot: Optional[dict] = None

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "workload_type": self.workload_type,
            "region": self.region,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "runtime_hours": round(self.runtime_hours, 3),
            "power_kw": round(self.power_kw, 2),
            "pue": round(self.pue, 3),
            "realized_energy_cost": round(self.realized_energy_cost, 4),
            "realized_carbon_kg": round(self.realized_carbon_kg, 4),
            "realized_sla_penalty": round(self.realized_sla_penalty, 4),
            "realized_data_transfer_cost": round(self.realized_data_transfer_cost, 6),
            "realized_total_cost": round(self.realized_total_cost, 4),
            "baseline_energy_cost": round(self.baseline_energy_cost, 4),
            "baseline_carbon_kg": round(self.baseline_carbon_kg, 4),
            "baseline_sla_penalty": round(self.baseline_sla_penalty, 4),
            "baseline_data_transfer_cost": round(self.baseline_data_transfer_cost, 6),
            "baseline_total_cost": round(self.baseline_total_cost, 4),
            "cost_savings": round(self.cost_savings, 4),
            "carbon_savings_kg": round(self.carbon_savings_kg, 4),
            "constraint_violations": self.constraint_violations,
            "forecast_snapshot": self.forecast_snapshot,
        }


@dataclass
class ShadowResult:
    """Aggregate result from a shadow evaluation run.

    Contains per-job records and aggregate metrics.
    All cost metrics are computed on realized price/carbon data — never forecasts.
    """
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.utcnow)

    records: list[ShadowDecisionRecord] = field(default_factory=list)
    constraint_violations: list[str] = field(default_factory=list)

    # Aggregate metrics (set by ShadowRunner.run())
    total_realized_cost: float = 0.0
    total_baseline_cost: float = 0.0
    total_realized_carbon_kg: float = 0.0
    total_baseline_carbon_kg: float = 0.0
    total_realized_sla_penalty: float = 0.0
    total_baseline_sla_penalty: float = 0.0

    @property
    def total_cost_savings(self) -> float:
        return self.total_baseline_cost - self.total_realized_cost

    @property
    def cost_savings_pct(self) -> float:
        if self.total_baseline_cost == 0:
            return 0.0
        return (self.total_cost_savings / self.total_baseline_cost) * 100

    @property
    def total_carbon_savings_kg(self) -> float:
        return self.total_baseline_carbon_kg - self.total_realized_carbon_kg

    @property
    def carbon_savings_pct(self) -> float:
        if self.total_baseline_carbon_kg == 0:
            return 0.0
        return (self.total_carbon_savings_kg / self.total_baseline_carbon_kg) * 100

    @property
    def jobs_evaluated(self) -> int:
        return len(self.records)

    @property
    def jobs_with_violations(self) -> int:
        return sum(1 for r in self.records if r.constraint_violations)

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "created_at": self.created_at.isoformat(),
            "jobs_evaluated": self.jobs_evaluated,
            "total_realized_cost": round(self.total_realized_cost, 4),
            "total_baseline_cost": round(self.total_baseline_cost, 4),
            "total_cost_savings": round(self.total_cost_savings, 4),
            "cost_savings_pct": round(self.cost_savings_pct, 2),
            "total_realized_carbon_kg": round(self.total_realized_carbon_kg, 4),
            "total_baseline_carbon_kg": round(self.total_baseline_carbon_kg, 4),
            "total_carbon_savings_kg": round(self.total_carbon_savings_kg, 4),
            "carbon_savings_pct": round(self.carbon_savings_pct, 2),
            "total_realized_sla_penalty": round(self.total_realized_sla_penalty, 4),
            "total_baseline_sla_penalty": round(self.total_baseline_sla_penalty, 4),
            "jobs_with_violations": self.jobs_with_violations,
            "constraint_violations": self.constraint_violations,
            "records": [r.to_dict() for r in self.records],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)


class ShadowRunner:
    """Replays Aurelius decisions against realized price/carbon data.

    This runner:
    1. Takes a list of ScheduleDecision objects (Aurelius output)
    2. Takes realized (actual) price and carbon time series
    3. Computes realized costs for each decision using real prices
    4. Computes baseline costs under the same realized prices
    5. Records everything without touching live infrastructure
    6. Returns a ShadowResult for reporting and learning loop

    Usage:
        runner = ShadowRunner()
        result = runner.run(
            decisions=aurelius_decisions,
            real_prices={"us-west": {t: price_per_mwh, ...}, ...},
            real_carbon={"us-west": {t: gco2_per_kwh, ...}, ...},
            jobs=jobs,
            baseline_decisions=baseline_decisions,
        )
    """

    def __init__(
        self,
        config: Optional[OptimizationConfig] = None,
        output_path: Optional[Path] = None,
    ):
        """
        Args:
            config: Optimization config used for cost weights.
            output_path: If set, write shadow results as JSONL to this path.
        """
        self.config = config or OptimizationConfig()
        self.objective_fn = ObjectiveFunction(self.config)
        self.output_path = output_path

    def run(
        self,
        decisions: list[ScheduleDecision],
        real_prices: dict[str, dict[datetime, float]],
        real_carbon: dict[str, dict[datetime, float]],
        jobs: list[Job],
        baseline_decisions: Optional[list[ScheduleDecision]] = None,
        run_id: Optional[str] = None,
    ) -> ShadowResult:
        """Replay decisions against realized prices/carbon.

        All cost computations use real_prices and real_carbon exclusively.
        Forecast data attached to decisions (decision.forecast) is stored
        for the learning loop but never used for cost computation here.

        Args:
            decisions: Aurelius scheduling decisions to evaluate.
            real_prices: Realized hourly prices {region: {hour_ts: $/MWh}}.
            real_carbon: Realized hourly carbon {region: {hour_ts: gCO2/kWh}}.
            jobs: Original job list (for metadata and SLA deadlines).
            baseline_decisions: Baseline schedule for comparison. If None,
                an ASAP baseline is constructed from job earliest_start.
            run_id: Optional run identifier. Auto-generated if not provided.

        Returns:
            ShadowResult with per-job records and aggregate metrics.
            Computed purely from realized data — no synthetic values.
        """
        result = ShadowResult(
            run_id=run_id or str(uuid.uuid4()),
        )

        job_map = {j.job_id: j for j in jobs}
        baseline_map: dict[str, ScheduleDecision] = {}

        if baseline_decisions:
            baseline_map = {d.job_id: d for d in baseline_decisions}
        else:
            # Construct ASAP baseline: start at earliest_start in first region
            for job in jobs:
                region = self.config.default_region if self.config.default_region in job.region_options else job.region_options[0]
                baseline_map[job.job_id] = ScheduleDecision(
                    job_id=job.job_id,
                    start_time=job.earliest_start,
                    region=region,
                    power_fraction=1.0,
                    actual_runtime_hours=job.runtime_hours,
                )

        global_violations: list[str] = []

        for decision in decisions:
            job = job_map.get(decision.job_id)
            if job is None:
                logger.warning("Shadow: job %s not found in job list", decision.job_id)
                global_violations.append(f"job_not_found:{decision.job_id}")
                continue

            baseline_decision = baseline_map.get(decision.job_id)

            # Validate data residency constraints
            violations = self._check_constraints(job, decision)
            if violations:
                global_violations.extend(violations)

            # Compute realized costs using ONLY real prices/carbon
            aurelius_obj = self.objective_fn.calculate(
                [job], [decision], real_prices, real_carbon
            )
            baseline_obj = self.objective_fn.calculate(
                [job], [baseline_decision], real_prices, real_carbon
            ) if baseline_decision else aurelius_obj

            record = ShadowDecisionRecord(
                job_id=decision.job_id,
                workload_type=getattr(job, "workload_type", "unknown"),
                region=decision.region,
                start_time=decision.start_time,
                end_time=decision.end_time,
                runtime_hours=decision.actual_runtime_hours,
                power_kw=job.power_kw * decision.power_fraction,
                pue=getattr(job, "pue", 1.0),
                realized_energy_cost=aurelius_obj.energy_cost,
                realized_carbon_kg=aurelius_obj.carbon_kg,
                realized_sla_penalty=aurelius_obj.sla_penalty_cost,
                realized_data_transfer_cost=aurelius_obj.data_transfer_cost,
                realized_total_cost=aurelius_obj.total,
                baseline_energy_cost=baseline_obj.energy_cost,
                baseline_carbon_kg=baseline_obj.carbon_kg,
                baseline_sla_penalty=baseline_obj.sla_penalty_cost,
                baseline_data_transfer_cost=baseline_obj.data_transfer_cost,
                baseline_total_cost=baseline_obj.total,
                cost_savings=baseline_obj.total - aurelius_obj.total,
                carbon_savings_kg=baseline_obj.carbon_kg - aurelius_obj.carbon_kg,
                constraint_violations=violations,
                forecast_snapshot=decision.forecast,
            )
            result.records.append(record)

            # Accumulate aggregates
            result.total_realized_cost += aurelius_obj.total
            result.total_baseline_cost += baseline_obj.total
            result.total_realized_carbon_kg += aurelius_obj.carbon_kg
            result.total_baseline_carbon_kg += baseline_obj.carbon_kg
            result.total_realized_sla_penalty += aurelius_obj.sla_penalty_cost
            result.total_baseline_sla_penalty += baseline_obj.sla_penalty_cost

        result.constraint_violations = global_violations

        logger.info(
            "Shadow run %s: %d jobs, cost_savings=%.4f (%.2f%%), "
            "carbon_savings_kg=%.2f, violations=%d",
            result.run_id,
            result.jobs_evaluated,
            result.total_cost_savings,
            result.cost_savings_pct,
            result.total_carbon_savings_kg,
            len(global_violations),
        )

        if self.output_path:
            self._persist(result)

        return result

    def _check_constraints(
        self,
        job: Job,
        decision: ScheduleDecision,
    ) -> list[str]:
        """Check scheduling constraints for a single decision.

        Returns list of violation strings (empty if no violations).
        """
        violations: list[str] = []

        # Data residency: decision region must not be forbidden
        forbidden = getattr(job, "forbidden_regions", [])
        if forbidden and decision.region in forbidden:
            violations.append(
                f"data_residency_violation:{job.job_id}:region={decision.region} is forbidden"
            )

        # Data residency: if allowed_regions set, decision must be in it
        allowed = getattr(job, "allowed_regions", [])
        if allowed and decision.region not in allowed:
            violations.append(
                f"data_residency_violation:{job.job_id}:region={decision.region} not in allowed={allowed}"
            )

        # Timing: job must not finish after deadline (if no SLA penalty)
        job_end = decision.start_time + timedelta(hours=decision.actual_runtime_hours)
        sla_penalty = getattr(job, "sla_penalty_per_hour", 0.0)
        if job_end > job.deadline and sla_penalty == 0.0:
            overrun = (job_end - job.deadline).total_seconds() / 3600
            violations.append(
                f"deadline_violation:{job.job_id}:overrun={overrun:.2f}h"
            )

        # Timing: must not start before earliest_start
        if decision.start_time < job.earliest_start:
            violations.append(
                f"earliest_start_violation:{job.job_id}:"
                f"start={decision.start_time} < earliest={job.earliest_start}"
            )

        # Region must be in job region_options
        if decision.region not in job.region_options:
            violations.append(
                f"invalid_region:{job.job_id}:region={decision.region} not in options={job.region_options}"
            )

        return violations

    def _persist(self, result: ShadowResult) -> None:
        """Write shadow result to JSONL output file."""
        try:
            path = Path(self.output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(result.to_json() + "\n")
            logger.info("Shadow result persisted to %s", path)
        except Exception as exc:
            logger.warning("Failed to persist shadow result: %s", exc)
