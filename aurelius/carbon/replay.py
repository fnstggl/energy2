"""Realized carbon replay + per-baseline savings (Phase 8).

Scores a schedule's CO2 against **historical** MOER (never forecast) using the
authoritative formula, and compares an optimized schedule against each baseline
schedule. Coverage is tracked and surfaced; there is **no silent 400 gCO2/kWh
fallback** — missing intervals lower coverage and are excluded from emissions,
so a low-coverage replay under-reports rather than fabricates.

For realized savings claims, feed this module HISTORICAL MOER. Feeding it
forecast MOER is allowed for diagnostics but the result's records will carry
``carbon_data_is_forecast=True`` so a report can refuse to call it "realized".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

from ..models import Job, ScheduleDecision
from .accounting import (
    CarbonDataKind,
    CarbonSignalType,
    MoerInterval,
    WorkloadCarbonRecord,
    aggregate_records,
    build_workload_carbon_record,
)


@dataclass
class ScheduleCarbonResult:
    """Carbon outcome for one schedule (optimized or a baseline)."""

    scheduler_name: str
    records: list[WorkloadCarbonRecord] = field(default_factory=list)

    @property
    def totals(self) -> dict:
        return aggregate_records(self.records)

    @property
    def total_emissions_kgco2(self) -> float:
        return self.totals["total_emissions_kgco2"]

    @property
    def total_energy_kwh(self) -> float:
        return self.totals["total_energy_kwh"]

    @property
    def emissions_intensity_gco2_per_kwh(self) -> float:
        return self.totals["emissions_intensity_gco2_per_kwh"]

    @property
    def carbon_data_coverage_pct(self) -> float:
        return self.totals["carbon_data_coverage_pct"]

    @property
    def carbon_data_is_real(self) -> bool:
        return self.totals["carbon_data_is_real"]

    def emissions_per_workload(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for r in self.records:
            out[r.job_id] = out.get(r.job_id, 0.0) + r.emissions_kgco2
        return out

    def to_dict(self) -> dict:
        t = self.totals
        return {
            "scheduler_name": self.scheduler_name,
            "total_emissions_kgco2": round(t["total_emissions_kgco2"], 6),
            "total_energy_kwh": round(t["total_energy_kwh"], 6),
            "emissions_intensity_gco2_per_kwh": round(t["emissions_intensity_gco2_per_kwh"], 4),
            "carbon_data_coverage_pct": round(t["carbon_data_coverage_pct"], 3),
            "carbon_data_is_real": t["carbon_data_is_real"],
            "carbon_missing_intervals": t["carbon_missing_intervals"],
            "n_workloads": len({r.job_id for r in self.records}),
        }


def _segment_moer_intervals(
    region: str,
    start: datetime,
    end: datetime,
    moer_data: dict[str, dict[datetime, float]],
    moer_is_real: bool,
) -> list[MoerInterval]:
    """Per-hour MOER readings over [start, end); None for missing (no default)."""
    region_moer = moer_data.get(region, {})
    intervals: list[MoerInterval] = []
    current = start.replace(minute=0, second=0, microsecond=0)
    while current < end:
        val = region_moer.get(current)
        intervals.append(MoerInterval(value_gco2_per_kwh=val, is_real=moer_is_real))
        current = current + timedelta(hours=1)
    return intervals


def replay_schedule_carbon(
    schedule: list[ScheduleDecision],
    jobs: list[Job],
    moer_data: dict[str, dict[datetime, float]],
    *,
    scheduler_name: str = "optimizer",
    carbon_data_source: str = "watttime_co2_moer",
    carbon_signal_type: CarbonSignalType = CarbonSignalType.MARGINAL_MOER,
    carbon_data_kind: CarbonDataKind = CarbonDataKind.HISTORICAL,
    moer_is_real: bool = True,
    default_pue: Optional[float] = None,
) -> ScheduleCarbonResult:
    """Compute realized carbon for a schedule using the authoritative formula.

    One :class:`WorkloadCarbonRecord` is produced per (job, segment). For a
    migrated job (multiple segments) callers can sum records via
    :func:`aurelius.carbon.accounting.aggregate_records` or
    ``ScheduleCarbonResult.emissions_per_workload``.

    PUE comes from ``Job.pue`` unless ``default_pue`` overrides it. Utilization
    is the segment's ``power_fraction`` (throttle level), matching how the
    evaluator/objective treat reduced-power runs.
    """
    job_by_id = {j.job_id: j for j in jobs}
    result = ScheduleCarbonResult(scheduler_name=scheduler_name)

    for decision in schedule:
        job = job_by_id.get(decision.job_id)
        if job is None:
            continue
        pue = default_pue if default_pue is not None else getattr(job, "pue", 1.0)
        for seg in decision.all_segments:
            intervals = _segment_moer_intervals(
                seg.region, seg.start_time, seg.end_time, moer_data, moer_is_real
            )
            if not intervals:
                continue
            record = build_workload_carbon_record(
                job_id=job.job_id,
                scheduler_name=scheduler_name,
                baseline_name=scheduler_name,
                region=seg.region,
                start_time_utc=seg.start_time,
                end_time_utc=seg.end_time,
                power_kw=job.power_kw,
                utilization_fraction=seg.power_fraction,
                pue=pue,
                moer_intervals=intervals,
                interval_hours=1.0,
                carbon_data_source=carbon_data_source,
                carbon_signal_type=carbon_signal_type,
                carbon_data_kind=carbon_data_kind,
            )
            result.records.append(record)

    return result


def _goodput_proxy(jobs: list[Job], schedule: list[ScheduleDecision]) -> float:
    """Goodput proxy = sum over scheduled jobs of gpu_count*runtime (or runtime).

    Mirrors the existing adapter's token_equivalent proxy so goodput_per_kgco2 is
    comparable across schedules.
    """
    job_by_id = {j.job_id: j for j in jobs}
    total = 0.0
    for d in schedule:
        job = job_by_id.get(d.job_id)
        if job is None:
            continue
        gc = max(0, getattr(job, "gpu_count", 0))
        total += (gc * job.runtime_hours) if gc > 0 else job.runtime_hours
    return total


def compare_baselines_carbon(
    optimized: ScheduleCarbonResult,
    baselines: dict[str, ScheduleCarbonResult],
    *,
    optimized_schedule: Optional[list[ScheduleDecision]] = None,
    baseline_schedules: Optional[dict[str, list[ScheduleDecision]]] = None,
    jobs: Optional[list[Job]] = None,
    optimized_cost_usd: Optional[float] = None,
    baseline_costs_usd: Optional[dict[str, float]] = None,
) -> dict:
    """Compare optimized emissions against each baseline.

    Produces, per baseline: total emissions, intensity, savings (kg and %), and —
    when schedules/jobs/costs are supplied — goodput, goodput_per_kgco2, and
    goodput_per_dollar. The comparison is marked ``carbon_data_is_real`` only when
    BOTH sides used real historical MOER with full coverage; otherwise it is
    flagged so a report labels it unverified rather than realized.
    """
    opt_emis = optimized.total_emissions_kgco2
    opt_goodput = (
        _goodput_proxy(jobs, optimized_schedule)
        if jobs is not None and optimized_schedule is not None else None
    )

    per_baseline: dict[str, dict] = {}
    for name, bl in baselines.items():
        bl_emis = bl.total_emissions_kgco2
        savings_kg = bl_emis - opt_emis
        savings_pct = (100.0 * savings_kg / bl_emis) if bl_emis > 0 else 0.0
        both_real = (
            optimized.carbon_data_is_real and bl.carbon_data_is_real
            and optimized.carbon_data_coverage_pct >= 100.0
            and bl.carbon_data_coverage_pct >= 100.0
        )
        entry = {
            "baseline_total_emissions_kgco2": round(bl_emis, 6),
            "optimized_total_emissions_kgco2": round(opt_emis, 6),
            "carbon_savings_kgco2_vs_baseline": round(savings_kg, 6),
            "carbon_savings_pct_vs_baseline": round(savings_pct, 4),
            "baseline_emissions_intensity_gco2_per_kwh": round(
                bl.emissions_intensity_gco2_per_kwh, 4),
            "optimized_emissions_intensity_gco2_per_kwh": round(
                optimized.emissions_intensity_gco2_per_kwh, 4),
            "carbon_savings_is_real_historical": both_real,
            "carbon_data_coverage_pct": round(
                min(optimized.carbon_data_coverage_pct, bl.carbon_data_coverage_pct), 3),
        }
        if baseline_costs_usd and optimized_cost_usd is not None and name in baseline_costs_usd:
            bl_cost = baseline_costs_usd[name]
            entry["baseline_cost_usd"] = round(bl_cost, 6)
            entry["optimized_cost_usd"] = round(optimized_cost_usd, 6)
            entry["cost_savings_usd_vs_baseline"] = round(bl_cost - optimized_cost_usd, 6)
        if opt_goodput is not None:
            entry["optimized_goodput"] = round(opt_goodput, 4)
            entry["optimized_goodput_per_kgco2"] = (
                round(opt_goodput / opt_emis, 6) if opt_emis > 0 else None
            )
            if optimized_cost_usd:
                entry["optimized_goodput_per_dollar"] = round(opt_goodput / optimized_cost_usd, 6)
        per_baseline[name] = entry

    return {
        "optimized": optimized.to_dict(),
        "per_baseline": per_baseline,
        "carbon_signal": "historical_moer" if (
            optimized.records and optimized.records[0].carbon_data_is_historical
        ) else "non_historical",
    }


__all__ = [
    "ScheduleCarbonResult",
    "replay_schedule_carbon",
    "compare_baselines_carbon",
]
