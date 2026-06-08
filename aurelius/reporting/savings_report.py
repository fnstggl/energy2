"""Savings report generator for Aurelius backtest results.

Generates reproducible savings reports with 95% bootstrap confidence intervals.
Compares the Aurelius optimizer against multiple baselines using strictly
realized (not forecast) price/carbon data from walk-forward backtest folds.

Leakage guarantee:
    All cost/carbon metrics are computed from BacktestRound.optimizer_metrics
    and BacktestRound.baseline_metrics, which are produced by evaluate_schedule()
    using only eval-window actuals. No forecast data enters these calculations.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np

from aurelius.backtesting.engine import BacktestRound
from aurelius.models import Job, ScheduleDecision


@dataclass
class ConfidenceInterval:
    """95% bootstrap confidence interval."""

    estimate: float
    lower_95: float
    upper_95: float
    n_samples: int

    def to_dict(self) -> dict:
        return {
            "estimate": round(self.estimate, 4),
            "lower_95": round(self.lower_95, 4),
            "upper_95": round(self.upper_95, 4),
            "n_bootstrap_samples": self.n_samples,
        }


def _bootstrap_ci(
    samples: list[float],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> ConfidenceInterval:
    """Compute 95% bootstrap CI by resampling samples with replacement."""
    rng = np.random.default_rng(seed)
    arr = np.array(samples, dtype=float)
    if len(arr) == 0:
        return ConfidenceInterval(
            estimate=float("nan"), lower_95=float("nan"),
            upper_95=float("nan"), n_samples=0,
        )
    if len(arr) == 1:
        return ConfidenceInterval(
            estimate=float(arr[0]), lower_95=float(arr[0]),
            upper_95=float(arr[0]), n_samples=1,
        )
    boot_means = []
    for _ in range(n_bootstrap):
        resample = rng.choice(arr, size=len(arr), replace=True)
        boot_means.append(float(resample.mean()))

    lower = float(np.percentile(boot_means, 2.5))
    upper = float(np.percentile(boot_means, 97.5))
    return ConfidenceInterval(
        estimate=float(arr.mean()),
        lower_95=lower,
        upper_95=upper,
        n_samples=n_bootstrap,
    )


def _latency_violations(
    schedule: list[ScheduleDecision],
    jobs: list[Job],
) -> int:
    """Count jobs whose completion time exceeds their deadline."""
    job_by_id = {j.job_id: j for j in jobs}
    violations = 0
    for dec in schedule:
        job = job_by_id.get(dec.job_id)
        if job is None:
            continue
        # Strip timezone for comparison (both normalized to naive UTC)
        start = dec.start_time
        if hasattr(start, "tzinfo") and start.tzinfo is not None:
            start = start.replace(tzinfo=None)
        completion = start + timedelta(hours=dec.actual_runtime_hours)
        deadline = job.deadline
        if hasattr(deadline, "tzinfo") and deadline.tzinfo is not None:
            deadline = deadline.replace(tzinfo=None)
        if completion > deadline:
            violations += 1
    return violations


def _queue_delay_hours(
    schedule: list[ScheduleDecision],
    jobs: list[Job],
) -> list[float]:
    """Return per-job scheduling delay (hours) from earliest_start to actual start."""
    job_by_id = {j.job_id: j for j in jobs}
    delays: list[float] = []
    for dec in schedule:
        job = job_by_id.get(dec.job_id)
        if job is None:
            continue
        earliest = job.earliest_start
        actual = dec.start_time
        if hasattr(earliest, "tzinfo") and earliest.tzinfo is not None:
            earliest = earliest.replace(tzinfo=None)
        if hasattr(actual, "tzinfo") and actual.tzinfo is not None:
            actual = actual.replace(tzinfo=None)
        delay = (actual - earliest).total_seconds() / 3600.0
        delays.append(max(0.0, delay))
    return delays


class SavingsReport:
    """Generate savings reports from walk-forward backtest results.

    Usage::
        rounds = BacktestEngine(...).run(jobs, price_df, carbon_df)
        report = SavingsReport.generate(rounds)
        html = render_html_report(report)
    """

    PRIMARY_BASELINE = "current_price_only"
    FALLBACK_BASELINES = ["fifo", "peak_blind_asap", "latency_first", "round_robin"]

    @classmethod
    def generate(
        cls,
        backtest_rounds: list[BacktestRound],
        n_bootstrap: int = 1000,
        primary_baseline: Optional[str] = None,
    ) -> dict:
        """Generate a savings report from backtest rounds.

        Args:
            backtest_rounds: List of BacktestRound from BacktestEngine.run().
            n_bootstrap: Bootstrap resamples for CI computation (default 1000).
            primary_baseline: Primary comparison baseline name.
                Defaults to "current_price_only", falling back to first available.

        Returns:
            dict with cost savings, carbon reduction, latency violations,
            utilization, queue delay, and 95% bootstrap CIs. All monetary
            values in USD; carbon in gCO2 and tonnes CO2.
        """
        if not backtest_rounds:
            return cls._empty_report()

        primary = primary_baseline or cls._select_primary_baseline(backtest_rounds)

        fold_data: list[dict] = []
        total_eval_jobs = 0
        total_latency_violations = 0
        total_jobs_evaluated = 0
        all_delays: list[float] = []
        carbon_hours_total = 0
        carbon_hours_covered = 0

        for round_ in backtest_rounds:
            if round_.optimizer_metrics is None:
                continue
            bl_metrics = round_.baseline_metrics.get(primary)
            if bl_metrics is None:
                continue

            opt = round_.optimizer_metrics
            bl = bl_metrics

            cost_savings_usd = bl.total_energy_cost_usd - opt.total_energy_cost_usd
            carbon_reduction_gco2 = bl.total_carbon_gco2 - opt.total_carbon_gco2
            cost_savings_pct = (
                100.0 * cost_savings_usd / bl.total_energy_cost_usd
                if bl.total_energy_cost_usd > 0
                else 0.0
            )
            carbon_reduction_pct = (
                100.0 * carbon_reduction_gco2 / bl.total_carbon_gco2
                if bl.total_carbon_gco2 > 0
                else 0.0
            )

            violations = _latency_violations(round_.optimizer_schedule, round_.eval_jobs)
            delays = _queue_delay_hours(round_.optimizer_schedule, round_.eval_jobs)

            total_eval_jobs += len(round_.eval_jobs)
            total_latency_violations += violations
            total_jobs_evaluated += opt.jobs_evaluated
            all_delays.extend(delays)
            # Carbon coverage (real MOER hours / total). Drives carbon_claim_validity.
            carbon_hours_total += getattr(opt, "carbon_hours_total", 0)
            carbon_hours_covered += getattr(opt, "carbon_hours_covered", 0)

            fold_data.append({
                "fold_index": round_.fold_index,
                "train_start": round_.train_start.isoformat(),
                "train_end": round_.train_end.isoformat(),
                "eval_start": round_.eval_start.isoformat(),
                "eval_end": round_.eval_end.isoformat(),
                "eval_jobs": len(round_.eval_jobs),
                "optimizer_cost_usd": opt.total_energy_cost_usd,
                "baseline_cost_usd": bl.total_energy_cost_usd,
                "cost_savings_usd": cost_savings_usd,
                "cost_savings_pct": cost_savings_pct,
                "optimizer_carbon_gco2": opt.total_carbon_gco2,
                "baseline_carbon_gco2": bl.total_carbon_gco2,
                "carbon_reduction_gco2": carbon_reduction_gco2,
                "carbon_reduction_pct": carbon_reduction_pct,
                "latency_violations": violations,
                "missing_price_hours": opt.missing_price_hours,
                "missing_carbon_hours": opt.missing_carbon_hours,
                "carbon_data_coverage_pct": getattr(opt, "carbon_data_coverage_pct", 0.0),
            })

        if not fold_data:
            return cls._empty_report()

        cost_savings_ci = _bootstrap_ci(
            [f["cost_savings_usd"] for f in fold_data], n_bootstrap
        )
        cost_pct_ci = _bootstrap_ci(
            [f["cost_savings_pct"] for f in fold_data], n_bootstrap
        )
        carbon_ci = _bootstrap_ci(
            [f["carbon_reduction_gco2"] for f in fold_data], n_bootstrap
        )
        carbon_pct_ci = _bootstrap_ci(
            [f["carbon_reduction_pct"] for f in fold_data], n_bootstrap
        )

        baseline_comparison = cls._build_baseline_comparison(backtest_rounds, n_bootstrap)

        # Carbon trust gate: a carbon saving may only be called "realized" when
        # the realized replay had full real MOER coverage. Otherwise it is
        # explicitly flagged so it cannot be presented as a real-world saving.
        carbon_coverage_pct = (
            100.0 * carbon_hours_covered / carbon_hours_total
            if carbon_hours_total else 0.0
        )
        if carbon_hours_total == 0:
            carbon_claim_validity = "no_carbon_data"
        elif carbon_coverage_pct >= 99.999:
            carbon_claim_validity = "realized_historical"
        else:
            carbon_claim_validity = "partial_coverage_unverified"

        total_opt_cost = sum(f["optimizer_cost_usd"] for f in fold_data)
        total_bl_cost = sum(f["baseline_cost_usd"] for f in fold_data)
        total_opt_carbon = sum(f["optimizer_carbon_gco2"] for f in fold_data)
        total_bl_carbon = sum(f["baseline_carbon_gco2"] for f in fold_data)
        total_cost_savings = total_bl_cost - total_opt_cost
        total_carbon_reduction = total_bl_carbon - total_opt_carbon

        utilization_pct = (
            100.0 * total_jobs_evaluated / total_eval_jobs if total_eval_jobs > 0 else 0.0
        )
        avg_delay = float(np.mean(all_delays)) if all_delays else 0.0
        median_delay = float(np.median(all_delays)) if all_delays else 0.0
        p95_delay = float(np.percentile(all_delays, 95)) if all_delays else 0.0

        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "n_folds": len(fold_data),
            "primary_baseline": primary,
            "totals": {
                "optimizer_cost_usd": round(total_opt_cost, 4),
                "baseline_cost_usd": round(total_bl_cost, 4),
                "cost_savings_usd": round(total_cost_savings, 4),
                "cost_savings_pct": round(
                    100.0 * total_cost_savings / total_bl_cost
                    if total_bl_cost > 0 else 0.0, 4
                ),
                "optimizer_carbon_gco2": round(total_opt_carbon, 4),
                "baseline_carbon_gco2": round(total_bl_carbon, 4),
                "carbon_reduction_gco2": round(total_carbon_reduction, 4),
                "carbon_reduction_tonnes": round(total_carbon_reduction / 1_000_000, 6),
                "carbon_reduction_pct": round(
                    100.0 * total_carbon_reduction / total_bl_carbon
                    if total_bl_carbon > 0 else 0.0, 4
                ),
                # Carbon trustworthiness — NEVER present carbon savings as real
                # unless this is "realized_historical".
                "carbon_data_coverage_pct": round(carbon_coverage_pct, 3),
                "carbon_claim_validity": carbon_claim_validity,
                "latency_violations": total_latency_violations,
                "latency_violation_rate_pct": round(
                    100.0 * total_latency_violations / total_eval_jobs
                    if total_eval_jobs > 0 else 0.0, 4
                ),
                "utilization_pct": round(utilization_pct, 4),
                "avg_queue_delay_hours": round(avg_delay, 4),
                "median_queue_delay_hours": round(median_delay, 4),
                "p95_queue_delay_hours": round(p95_delay, 4),
            },
            "confidence_intervals": {
                "cost_savings_usd_per_fold": cost_savings_ci.to_dict(),
                "cost_savings_pct_per_fold": cost_pct_ci.to_dict(),
                "carbon_reduction_gco2_per_fold": carbon_ci.to_dict(),
                "carbon_reduction_pct_per_fold": carbon_pct_ci.to_dict(),
            },
            "baseline_comparison": baseline_comparison,
            "fold_results": fold_data,
            "methodology": cls._methodology_section(),
        }

    @classmethod
    def _select_primary_baseline(cls, rounds: list[BacktestRound]) -> str:
        available: set[str] = set()
        for r in rounds:
            available.update(r.baseline_metrics.keys())
        if cls.PRIMARY_BASELINE in available:
            return cls.PRIMARY_BASELINE
        for name in cls.FALLBACK_BASELINES:
            if name in available:
                return name
        return next(iter(available)) if available else "unknown"

    @classmethod
    def _build_baseline_comparison(
        cls,
        rounds: list[BacktestRound],
        n_bootstrap: int,
    ) -> dict:
        all_names: set[str] = set()
        for r in rounds:
            all_names.update(r.baseline_metrics.keys())

        comparison: dict[str, dict] = {}
        for name in sorted(all_names):
            savings_samples: list[float] = []
            carbon_savings_samples: list[float] = []
            carbon_complete = True
            for r in rounds:
                if r.optimizer_metrics is None:
                    continue
                bl = r.baseline_metrics.get(name)
                if bl is None:
                    continue
                opt = r.optimizer_metrics
                if bl.total_energy_cost_usd > 0:
                    savings_samples.append(
                        100.0 * (bl.total_energy_cost_usd - opt.total_energy_cost_usd)
                        / bl.total_energy_cost_usd
                    )
                # Per-baseline carbon savings (realized historical MOER).
                if bl.total_carbon_gco2 > 0:
                    carbon_savings_samples.append(
                        100.0 * (bl.total_carbon_gco2 - opt.total_carbon_gco2)
                        / bl.total_carbon_gco2
                    )
                if not (getattr(opt, "carbon_complete", False)
                        and getattr(bl, "carbon_complete", False)):
                    carbon_complete = False
            ci = _bootstrap_ci(savings_samples, n_bootstrap)
            carbon_ci = _bootstrap_ci(carbon_savings_samples, n_bootstrap)
            comparison[name] = {
                "cost_savings_pct": ci.to_dict(),
                "carbon_savings_pct": carbon_ci.to_dict(),
                "carbon_claim_validity": (
                    "realized_historical" if (carbon_savings_samples and carbon_complete)
                    else "partial_coverage_unverified" if carbon_savings_samples
                    else "no_carbon_data"
                ),
                "n_folds": len(savings_samples),
            }
        return comparison

    @classmethod
    def _empty_report(cls) -> dict:
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "n_folds": 0,
            "primary_baseline": "none",
            "totals": {},
            "confidence_intervals": {},
            "baseline_comparison": {},
            "fold_results": [],
            "methodology": cls._methodology_section(),
            "warning": (
                "No backtest rounds produced metrics. "
                "Verify price/carbon data and job windows."
            ),
        }

    @staticmethod
    def _methodology_section() -> dict:
        return {
            "leakage_free_guarantee": (
                "All cost and carbon metrics are computed using evaluate_schedule(), "
                "which uses only eval-window realized prices and carbon data. "
                "Forecast data is never used in cost/carbon calculations. "
                "The BacktestEngine enforces train_end < eval_start on every fold "
                "via TemporalSplitter, and assert_no_leakage() validates this "
                "invariant before each fold executes."
            ),
            "confidence_intervals": (
                "95% confidence intervals are computed by bootstrap resampling "
                "(n=1000) across backtest folds. Each bootstrap sample draws folds "
                "with replacement and computes the mean savings. The 2.5th and "
                "97.5th percentiles define the lower and upper CI bounds."
            ),
            "primary_baseline": (
                "The primary comparison baseline is 'current_price_only', which "
                "schedules each job in the cheapest available region at the current "
                "observed price without forecasting. This is the most natural "
                "comparison: a naive operator who always picks the cheapest region now."
            ),
            "carbon_unit": (
                "Carbon in gCO2 (grams of CO2 equivalent). "
                "Conversion to tonnes: divide by 1,000,000."
            ),
            "cost_unit": (
                "USD (US Dollars). Energy cost = sum over job-hours of "
                "(price_per_mwh / 1000) * power_kw * pue_factor."
            ),
            "reproduction": (
                "To reproduce: obtain an EIA API key or use CSV-imported historical "
                "price/carbon data, then run: "
                "'aurelius backtest --start YYYY-MM-DD --end YYYY-MM-DD --region <region>'. "
                "All decisions and outcomes are deterministic given the same data."
            ),
        }
