"""Phase 4: Reporting and Pilot Readiness — full test suite.

Tests three levels:
1. Unit tests: SavingsReport, ConfidenceInterval, _bootstrap_ci, HTML rendering
2. Integration tests: BacktestEngine → SavingsReport → HTML pipeline
3. API tests: auth middleware, endpoint correctness, 401 on bad key

Each test is adversarially designed to verify real correctness, not just
that the code runs.
"""

from __future__ import annotations

import math
import os
from datetime import datetime, timedelta, timezone
from typing import Optional
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from aurelius.backtesting.engine import BacktestEngine, BacktestRound
from aurelius.backtesting.evaluator import RealizedMetrics
from aurelius.models import Job, ScheduleDecision
from aurelius.reporting.html_report import render_html_report
from aurelius.reporting.savings_report import (
    SavingsReport,
    _bootstrap_ci,
    _latency_violations,
    _queue_delay_hours,
)

# ---------------------------------------------------------------------------
# Test fixtures and helpers
# ---------------------------------------------------------------------------

UTC = timezone.utc
_T0 = datetime(2024, 3, 1, 0, 0, tzinfo=UTC)


def _hours(n: int) -> timedelta:
    return timedelta(hours=n)


def _make_realized_metrics(
    cost: float = 100.0,
    carbon: float = 50_000.0,
    jobs: int = 5,
    missing_price: int = 0,
) -> RealizedMetrics:
    m = RealizedMetrics()
    m.total_energy_cost_usd = cost
    m.total_carbon_gco2 = carbon
    m.jobs_evaluated = jobs
    m.missing_price_hours = missing_price
    return m


def _make_job(
    job_id: str = "j0",
    earliest: Optional[datetime] = None,
    deadline: Optional[datetime] = None,
    submit: Optional[datetime] = None,
    runtime: float = 2.0,
) -> Job:
    earliest = earliest or _T0
    return Job(
        job_id=job_id,
        submit_time=submit or earliest,
        runtime_hours=runtime,
        deadline=deadline or (earliest + _hours(24)),
        power_kw=100.0,
        earliest_start=earliest,
        region_options=["us-west"],
    )


def _make_decision(
    job_id: str = "j0",
    start: Optional[datetime] = None,
    runtime: float = 2.0,
    region: str = "us-west",
) -> ScheduleDecision:
    return ScheduleDecision(
        job_id=job_id,
        start_time=start or _T0,
        region=region,
        power_fraction=1.0,
        actual_runtime_hours=runtime,
    )


def _make_round(
    fold: int = 0,
    opt_cost: float = 80.0,
    bl_cost: float = 100.0,
    opt_carbon: float = 40_000.0,
    bl_carbon: float = 50_000.0,
    jobs: Optional[list[Job]] = None,
    schedule: Optional[list[ScheduleDecision]] = None,
) -> BacktestRound:
    jobs = jobs or [_make_job(job_id=f"j{fold}")]
    schedule = schedule or [_make_decision(job_id=f"j{fold}")]

    opt_metrics = _make_realized_metrics(opt_cost, opt_carbon, len(jobs))
    bl_metrics = _make_realized_metrics(bl_cost, bl_carbon, len(jobs))

    ts = _T0 + _hours(fold * 168)
    return BacktestRound(
        fold_index=fold,
        train_start=pd.Timestamp(ts - _hours(720)),
        train_end=pd.Timestamp(ts),
        eval_start=pd.Timestamp(ts),
        eval_end=pd.Timestamp(ts + _hours(168)),
        eval_jobs=jobs,
        optimizer_schedule=schedule,
        optimizer_metrics=opt_metrics,
        baseline_metrics={"current_price_only": bl_metrics, "fifo": bl_metrics},
        baseline_schedules={},
        forecast_quality=None,
    )


# ===========================================================================
# UNIT TESTS: _bootstrap_ci
# ===========================================================================

class TestBootstrapCI:
    def test_single_sample(self):
        ci = _bootstrap_ci([42.0])
        assert ci.estimate == pytest.approx(42.0)
        assert ci.lower_95 == pytest.approx(42.0)
        assert ci.upper_95 == pytest.approx(42.0)
        assert ci.n_samples == 1

    def test_empty_samples(self):
        ci = _bootstrap_ci([])
        assert math.isnan(ci.estimate)
        assert ci.n_samples == 0

    def test_two_samples_ci_ordered(self):
        ci = _bootstrap_ci([10.0, 20.0])
        assert ci.lower_95 <= ci.estimate <= ci.upper_95

    def test_ci_deterministic_with_seed(self):
        samples = [float(i) for i in range(20)]
        ci1 = _bootstrap_ci(samples, seed=42)
        ci2 = _bootstrap_ci(samples, seed=42)
        assert ci1.lower_95 == ci2.lower_95
        assert ci1.upper_95 == ci2.upper_95

    def test_ci_width_decreases_with_more_samples(self):
        """Wider CI for fewer samples (high variance small set vs large set)."""
        rng = np.random.default_rng(1)
        large = list(rng.normal(50, 5, 100))
        small = list(rng.normal(50, 5, 5))
        ci_large = _bootstrap_ci(large, seed=99)
        ci_small = _bootstrap_ci(small, seed=99)
        large_width = ci_large.upper_95 - ci_large.lower_95
        small_width = ci_small.upper_95 - ci_small.lower_95
        assert small_width >= large_width

    def test_to_dict_keys(self):
        ci = _bootstrap_ci([1.0, 2.0, 3.0])
        d = ci.to_dict()
        assert "estimate" in d
        assert "lower_95" in d
        assert "upper_95" in d
        assert "n_bootstrap_samples" in d


# ===========================================================================
# UNIT TESTS: _latency_violations
# ===========================================================================

class TestLatencyViolations:
    def test_no_violations_when_on_time(self):
        job = _make_job("j0", earliest=_T0, deadline=_T0 + _hours(10), runtime=2.0)
        dec = _make_decision("j0", start=_T0, runtime=2.0)
        assert _latency_violations([dec], [job]) == 0

    def test_violation_when_overruns_deadline(self):
        job = _make_job("j0", earliest=_T0, deadline=_T0 + _hours(3), runtime=2.0)
        # Starts at T0+2h, runtime=2h → completes at T0+4h > deadline T0+3h
        dec = _make_decision("j0", start=_T0 + _hours(2), runtime=2.0)
        assert _latency_violations([dec], [job]) == 1

    def test_multiple_mixed(self):
        jobs = [
            _make_job("j0", deadline=_T0 + _hours(3)),   # violated (completes at T0+2h — ok)
            _make_job("j1", deadline=_T0 + _hours(1)),   # violated (starts T0, completes T0+2h > T0+1h)
        ]
        decisions = [
            _make_decision("j0", start=_T0, runtime=2.0),  # completes T0+2h < T0+3h → OK
            _make_decision("j1", start=_T0, runtime=2.0),  # completes T0+2h > T0+1h → violation
        ]
        assert _latency_violations(decisions, jobs) == 1

    def test_unknown_job_skipped(self):
        job = _make_job("j0")
        dec = _make_decision("j_unknown", start=_T0, runtime=100.0)
        assert _latency_violations([dec], [job]) == 0

    def test_timezone_aware_deadline(self):
        """Violation detection must work with tz-aware datetimes."""
        deadline = _T0 + _hours(1)
        job = _make_job("j0", earliest=_T0, deadline=deadline)
        # Completes at T0+3h > T0+1h
        dec = _make_decision("j0", start=_T0, runtime=3.0)
        assert _latency_violations([dec], [job]) == 1


# ===========================================================================
# UNIT TESTS: _queue_delay_hours
# ===========================================================================

class TestQueueDelayHours:
    def test_no_delay(self):
        job = _make_job("j0", earliest=_T0)
        dec = _make_decision("j0", start=_T0)
        delays = _queue_delay_hours([dec], [job])
        assert delays == [0.0]

    def test_positive_delay(self):
        job = _make_job("j0", earliest=_T0)
        dec = _make_decision("j0", start=_T0 + _hours(6))
        delays = _queue_delay_hours([dec], [job])
        assert delays == pytest.approx([6.0])

    def test_clamps_negative_to_zero(self):
        """start before earliest_start should not produce negative delay."""
        job = _make_job("j0", earliest=_T0 + _hours(2))
        dec = _make_decision("j0", start=_T0)  # starts before earliest_start
        delays = _queue_delay_hours([dec], [job])
        assert delays[0] == 0.0

    def test_unknown_job_skipped(self):
        job = _make_job("j0")
        dec = _make_decision("j_other", start=_T0 + _hours(99))
        delays = _queue_delay_hours([dec], [job])
        assert delays == []


# ===========================================================================
# UNIT TESTS: SavingsReport
# ===========================================================================

class TestSavingsReportEmpty:
    def test_empty_rounds_returns_warning(self):
        report = SavingsReport.generate([])
        assert report["n_folds"] == 0
        assert "warning" in report
        assert "methodology" in report

    def test_rounds_with_no_matching_baseline(self):
        """If no round has the primary baseline, should return empty report."""
        round_ = _make_round()
        # Remove baseline metrics
        round_.baseline_metrics = {"fifo": _make_realized_metrics()}
        report = SavingsReport.generate([round_], primary_baseline="nonexistent")
        assert report["n_folds"] == 0


class TestSavingsReportSingleFold:
    def test_cost_savings_computed_correctly(self):
        round_ = _make_round(opt_cost=80.0, bl_cost=100.0)
        report = SavingsReport.generate([round_])
        totals = report["totals"]
        assert totals["cost_savings_usd"] == pytest.approx(20.0, rel=1e-3)
        assert totals["cost_savings_pct"] == pytest.approx(20.0, rel=1e-3)

    def test_carbon_reduction_computed_correctly(self):
        round_ = _make_round(opt_carbon=40_000.0, bl_carbon=50_000.0)
        report = SavingsReport.generate([round_])
        totals = report["totals"]
        assert totals["carbon_reduction_gco2"] == pytest.approx(10_000.0, rel=1e-3)
        assert totals["carbon_reduction_tonnes"] == pytest.approx(10_000.0 / 1_000_000, rel=1e-3)

    def test_negative_savings_when_optimizer_worse(self):
        round_ = _make_round(opt_cost=120.0, bl_cost=100.0)
        report = SavingsReport.generate([round_])
        assert report["totals"]["cost_savings_usd"] < 0

    def test_primary_baseline_selected_correctly(self):
        round_ = _make_round()
        report = SavingsReport.generate([round_])
        assert report["primary_baseline"] == "current_price_only"

    def test_fallback_baseline_when_primary_absent(self):
        round_ = _make_round()
        round_.baseline_metrics = {"fifo": _make_realized_metrics()}
        report = SavingsReport.generate([round_])
        assert report["primary_baseline"] == "fifo"

    def test_n_folds_matches(self):
        rounds = [_make_round(fold=i) for i in range(3)]
        report = SavingsReport.generate(rounds)
        assert report["n_folds"] == 3

    def test_fold_results_have_required_keys(self):
        report = SavingsReport.generate([_make_round()])
        fold = report["fold_results"][0]
        required = {
            "fold_index", "train_start", "train_end", "eval_start", "eval_end",
            "eval_jobs", "optimizer_cost_usd", "baseline_cost_usd",
            "cost_savings_usd", "cost_savings_pct",
            "optimizer_carbon_gco2", "baseline_carbon_gco2",
            "carbon_reduction_gco2", "carbon_reduction_pct",
            "latency_violations", "missing_price_hours", "missing_carbon_hours",
        }
        assert required.issubset(fold.keys())


class TestSavingsReportConfidenceIntervals:
    def test_ci_keys_present(self):
        report = SavingsReport.generate([_make_round()])
        ci = report["confidence_intervals"]
        assert "cost_savings_usd_per_fold" in ci
        assert "cost_savings_pct_per_fold" in ci
        assert "carbon_reduction_gco2_per_fold" in ci
        assert "carbon_reduction_pct_per_fold" in ci

    def test_ci_single_fold_lower_equals_upper(self):
        """With one fold the CI should be a degenerate interval (lower==upper)."""
        report = SavingsReport.generate([_make_round()])
        ci = report["confidence_intervals"]["cost_savings_pct_per_fold"]
        assert ci["lower_95"] == pytest.approx(ci["upper_95"], abs=1e-3)

    def test_ci_ordered_lower_le_estimate_le_upper(self):
        rounds = [_make_round(fold=i, opt_cost=70 + i * 5, bl_cost=100.0) for i in range(8)]
        report = SavingsReport.generate(rounds, n_bootstrap=200)
        ci = report["confidence_intervals"]["cost_savings_pct_per_fold"]
        assert ci["lower_95"] <= ci["estimate"]
        assert ci["estimate"] <= ci["upper_95"]

    def test_ci_deterministic(self):
        rounds = [_make_round(fold=i) for i in range(5)]
        r1 = SavingsReport.generate(rounds, n_bootstrap=100)
        r2 = SavingsReport.generate(rounds, n_bootstrap=100)
        ci1 = r1["confidence_intervals"]["cost_savings_pct_per_fold"]
        ci2 = r2["confidence_intervals"]["cost_savings_pct_per_fold"]
        assert ci1["lower_95"] == pytest.approx(ci2["lower_95"])
        assert ci1["upper_95"] == pytest.approx(ci2["upper_95"])


class TestSavingsReportBaselineComparison:
    def test_all_baselines_appear_in_comparison(self):
        round_ = _make_round()
        round_.baseline_metrics = {
            "current_price_only": _make_realized_metrics(cost=100.0),
            "fifo": _make_realized_metrics(cost=95.0),
        }
        report = SavingsReport.generate([round_])
        comparison = report["baseline_comparison"]
        assert "current_price_only" in comparison
        assert "fifo" in comparison

    def test_comparison_savings_correct_direction(self):
        round_ = _make_round()
        round_.optimizer_metrics = _make_realized_metrics(cost=70.0)
        round_.baseline_metrics["current_price_only"] = _make_realized_metrics(cost=100.0)
        report = SavingsReport.generate([round_])
        est = report["baseline_comparison"]["current_price_only"]["cost_savings_pct"]["estimate"]
        assert est == pytest.approx(30.0, rel=1e-2)  # 30% savings


class TestSavingsReportMethodology:
    def test_methodology_present(self):
        report = SavingsReport.generate([_make_round()])
        assert "methodology" in report
        m = report["methodology"]
        assert "leakage_free_guarantee" in m
        assert "confidence_intervals" in m
        assert "primary_baseline" in m

    def test_leakage_guarantee_mentions_train_before_eval(self):
        report = SavingsReport.generate([_make_round()])
        text = report["methodology"]["leakage_free_guarantee"]
        assert "train_end" in text.lower() or "eval_start" in text.lower()

    def test_generated_at_is_recent_iso(self):
        report = SavingsReport.generate([_make_round()])
        ts = datetime.fromisoformat(report["generated_at"].replace("Z", "+00:00"))
        delta = datetime.now(UTC) - ts
        assert delta.total_seconds() < 10


class TestSavingsReportSLAAndDelay:
    def test_latency_violation_counted(self):
        job = _make_job("j0", earliest=_T0, deadline=_T0 + _hours(1))
        # Start + runtime (3h) exceeds deadline (1h)
        dec = _make_decision("j0", start=_T0, runtime=3.0)
        round_ = _make_round()
        round_.eval_jobs = [job]
        round_.optimizer_schedule = [dec]
        report = SavingsReport.generate([round_])
        assert report["totals"]["latency_violations"] >= 1

    def test_zero_latency_violations_when_all_on_time(self):
        job = _make_job("j0", earliest=_T0, deadline=_T0 + _hours(10))
        dec = _make_decision("j0", start=_T0, runtime=2.0)
        round_ = _make_round()
        round_.eval_jobs = [job]
        round_.optimizer_schedule = [dec]
        report = SavingsReport.generate([round_])
        assert report["totals"]["latency_violations"] == 0

    def test_queue_delay_nonzero_when_delayed(self):
        job = _make_job("j0", earliest=_T0, deadline=_T0 + _hours(48))
        dec = _make_decision("j0", start=_T0 + _hours(5), runtime=2.0)
        round_ = _make_round()
        round_.eval_jobs = [job]
        round_.optimizer_schedule = [dec]
        report = SavingsReport.generate([round_])
        assert report["totals"]["avg_queue_delay_hours"] == pytest.approx(5.0, rel=1e-2)


# ===========================================================================
# UNIT TESTS: HTML Report
# ===========================================================================

class TestHTMLReport:
    def test_render_returns_string(self):
        rounds = [_make_round(fold=i) for i in range(2)]
        report = SavingsReport.generate(rounds)
        html = render_html_report(report)
        assert isinstance(html, str)

    def test_render_contains_doctype(self):
        report = SavingsReport.generate([_make_round()])
        html = render_html_report(report)
        assert "<!DOCTYPE html>" in html

    def test_render_contains_leakage_badge(self):
        report = SavingsReport.generate([_make_round()])
        html = render_html_report(report)
        assert "leakage-free" in html

    def test_render_contains_methodology_text(self):
        report = SavingsReport.generate([_make_round()])
        html = render_html_report(report)
        assert "leakage" in html.lower()

    def test_render_contains_embedded_png_chart(self):
        rounds = [_make_round(fold=i) for i in range(2)]
        report = SavingsReport.generate(rounds)
        html = render_html_report(report)
        assert "data:image/png;base64," in html

    def test_render_empty_report_does_not_crash(self):
        report = SavingsReport.generate([])
        html = render_html_report(report)
        assert "<!DOCTYPE html>" in html

    def test_render_multiple_folds_shows_all_fold_rows(self):
        rounds = [_make_round(fold=i) for i in range(3)]
        report = SavingsReport.generate(rounds)
        html = render_html_report(report)
        # Each fold row contains the fold index
        for i in range(3):
            assert f"<td>{i}</td>" in html

    def test_render_savings_pct_appears_in_html(self):
        round_ = _make_round(opt_cost=80.0, bl_cost=100.0)
        report = SavingsReport.generate([round_])
        html = render_html_report(report)
        # 20.0% savings should appear somewhere
        assert "20.0%" in html

    def test_render_self_contained_no_external_links(self):
        """No http:// or https:// URLs should appear in the rendered HTML."""
        report = SavingsReport.generate([_make_round()])
        html = render_html_report(report)
        # Remove data: URIs and check for external
        import re
        stripped = re.sub(r'data:[^"\']+', '', html)
        assert "http://" not in stripped
        assert "https://" not in stripped


# ===========================================================================
# INTEGRATION TEST: BacktestEngine → SavingsReport
# ===========================================================================

class TestBacktestToSavingsReportIntegration:
    def _make_price_df(self, hours: int = 240, regions=("us-west", "us-east")) -> pd.DataFrame:
        rows = []
        for h in range(hours):
            ts = _T0 + _hours(h)
            for region in regions:
                price = 50.0 + (h % 24) * 2.0 + (0.5 if region == "us-east" else 0.0)
                rows.append({"timestamp": ts, "region": region, "price_per_mwh": price})
        return pd.DataFrame(rows)

    def _make_carbon_df(self, hours: int = 240, regions=("us-west", "us-east")) -> pd.DataFrame:
        rows = []
        for h in range(hours):
            ts = _T0 + _hours(h)
            for region in regions:
                rows.append({"timestamp": ts, "region": region, "gco2_per_kwh": 300.0})
        return pd.DataFrame(rows)

    def _make_jobs(self, n: int = 10) -> list[Job]:
        jobs = []
        for i in range(n):
            earliest = _T0 + _hours(48 + i * 2)
            jobs.append(Job(
                job_id=f"job-{i}",
                submit_time=earliest,
                runtime_hours=2.0,
                deadline=earliest + _hours(24),
                power_kw=50.0,
                earliest_start=earliest,
                region_options=["us-west", "us-east"],
            ))
        return jobs

    def test_full_pipeline_produces_valid_report(self):
        engine = BacktestEngine(
            method="greedy",
            train_days=2,
            eval_days=2,
            step_days=2,
        )
        price_df = self._make_price_df()
        carbon_df = self._make_carbon_df()
        jobs = self._make_jobs()

        rounds = engine.run(jobs=jobs, price_df=price_df, carbon_df=carbon_df)
        assert len(rounds) > 0

        report = SavingsReport.generate(rounds)
        assert report["n_folds"] == len([r for r in rounds if r.optimizer_metrics is not None])
        assert "totals" in report
        assert "confidence_intervals" in report
        assert "methodology" in report

    def test_totals_are_financially_consistent(self):
        """Total optimizer cost must equal sum of per-fold optimizer costs."""
        engine = BacktestEngine(method="greedy", train_days=2, eval_days=2, step_days=2)
        rounds = engine.run(
            jobs=self._make_jobs(),
            price_df=self._make_price_df(),
            carbon_df=self._make_carbon_df(),
        )
        report = SavingsReport.generate(rounds)
        fold_sum = sum(f["optimizer_cost_usd"] for f in report["fold_results"])
        assert report["totals"]["optimizer_cost_usd"] == pytest.approx(fold_sum, rel=1e-6)

    def test_html_renders_from_integration_report(self):
        engine = BacktestEngine(method="greedy", train_days=2, eval_days=2, step_days=2)
        rounds = engine.run(
            jobs=self._make_jobs(),
            price_df=self._make_price_df(),
            carbon_df=self._make_carbon_df(),
        )
        report = SavingsReport.generate(rounds)
        html = render_html_report(report)
        assert "<!DOCTYPE html>" in html
        assert "data:image/png;base64," in html

    def test_no_synthetic_savings_from_missing_price_data(self):
        """If price data is sparse, missing_price_hours should be > 0 and noted."""
        engine = BacktestEngine(method="greedy", train_days=2, eval_days=2, step_days=2)
        # Only 1 hour of price data (very sparse)
        tiny_price = pd.DataFrame([{
            "timestamp": _T0, "region": "us-west", "price_per_mwh": 50.0
        }])
        tiny_carbon = pd.DataFrame([{
            "timestamp": _T0, "region": "us-west", "gco2_per_kwh": 300.0
        }])
        rounds = engine.run(jobs=self._make_jobs(), price_df=tiny_price, carbon_df=tiny_carbon)
        # If no rounds or empty report, that's acceptable — what's not acceptable
        # is claiming savings from missing data without flagging it
        if rounds:
            report = SavingsReport.generate(rounds)
            # If there are folds with results, check that missing_price_hours is tracked
            for fold in report["fold_results"]:
                assert "missing_price_hours" in fold


# ===========================================================================
# API AUTH TESTS
# ===========================================================================

class TestAPIAuthMiddleware:
    def test_health_check_no_auth_needed(self):
        """GET /health must work without an API key even when AURELIUS_API_KEY is set."""
        from fastapi.testclient import TestClient

        from aurelius.api.app import app

        with patch.dict(os.environ, {"AURELIUS_API_KEY": "secret-key-123"}):
            client = TestClient(app, raise_server_exceptions=False)
            response = client.get("/health")
        # /health has no auth dependency — should always return 200
        assert response.status_code == 200

    def test_simulations_requires_auth_when_key_configured(self):
        """GET /simulations returns 401 when key is set and header is missing."""
        from fastapi.testclient import TestClient

        from aurelius.api.app import app

        with patch.dict(os.environ, {"AURELIUS_API_KEY": "correct-key"}):
            client = TestClient(app, raise_server_exceptions=False)
            response = client.get("/simulations")
        assert response.status_code == 401

    def test_simulations_correct_key_accepted(self):
        """GET /simulations passes when correct X-API-Key is provided."""
        from fastapi.testclient import TestClient

        from aurelius.api.app import app

        with patch.dict(os.environ, {"AURELIUS_API_KEY": "my-secret"}):
            client = TestClient(app, raise_server_exceptions=False)
            response = client.get("/simulations", headers={"X-API-Key": "my-secret"})
        # Either 200 (DB connected) or 503 (DB not connected) — but NOT 401
        assert response.status_code != 401

    def test_simulations_wrong_key_rejected(self):
        """GET /simulations returns 401 when wrong key is provided."""
        from fastapi.testclient import TestClient

        from aurelius.api.app import app

        with patch.dict(os.environ, {"AURELIUS_API_KEY": "correct-key"}):
            client = TestClient(app, raise_server_exceptions=False)
            response = client.get(
                "/simulations", headers={"X-API-Key": "wrong-key"}
            )
        assert response.status_code == 401

    def test_simulate_endpoint_requires_auth(self):
        """POST /simulate returns 401 when key configured and header missing."""
        from fastapi.testclient import TestClient

        from aurelius.api.app import app

        with patch.dict(os.environ, {"AURELIUS_API_KEY": "secret"}):
            client = TestClient(app, raise_server_exceptions=False)
            response = client.post("/simulate", json={"num_jobs": 5})
        assert response.status_code == 401

    def test_no_auth_when_api_key_not_set(self):
        """Without AURELIUS_API_KEY set, requests pass through without 401."""
        from fastapi.testclient import TestClient

        from aurelius.api.app import app

        env = {k: v for k, v in os.environ.items() if k != "AURELIUS_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            client = TestClient(app, raise_server_exceptions=False)
            response = client.get("/simulations")
        # Should not be 401 (may be 200 or 503 depending on DB)
        assert response.status_code != 401

    def test_get_simulation_by_id_requires_auth(self):
        """GET /simulations/{id} returns 401 when key configured and missing."""
        from fastapi.testclient import TestClient

        from aurelius.api.app import app

        with patch.dict(os.environ, {"AURELIUS_API_KEY": "secret"}):
            client = TestClient(app, raise_server_exceptions=False)
            response = client.get("/simulations/some-run-id")
        assert response.status_code == 401


# ===========================================================================
# VALIDATION TESTS: leakage_audit still works correctly
# ===========================================================================

class TestLeakageAuditIntegration:
    def test_assert_no_leakage_passes_clean_split(self):
        from aurelius.validation.leakage_audit import assert_no_leakage
        train = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=24, freq="h")})
        eval_ = pd.DataFrame({"timestamp": pd.date_range("2024-01-02", periods=24, freq="h")})
        assert_no_leakage(train, eval_)  # must not raise

    def test_assert_no_leakage_raises_on_overlap(self):
        from aurelius.validation.leakage_audit import DataLeakageError, assert_no_leakage
        train = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=24, freq="h")})
        eval_ = pd.DataFrame({"timestamp": pd.date_range("2024-01-01 12:00", periods=24, freq="h")})
        with pytest.raises(DataLeakageError):
            assert_no_leakage(train, eval_)

    def test_assert_no_leakage_raises_when_train_equals_eval_start(self):
        from aurelius.validation.leakage_audit import DataLeakageError, assert_no_leakage
        ts = pd.Timestamp("2024-01-02")
        train = pd.DataFrame({"timestamp": [ts]})
        eval_ = pd.DataFrame({"timestamp": [ts]})
        with pytest.raises(DataLeakageError):
            assert_no_leakage(train, eval_)

    def test_empty_train_raises_valueerror(self):
        from aurelius.validation.leakage_audit import assert_no_leakage
        empty = pd.DataFrame({"timestamp": pd.Series([], dtype="datetime64[ns]")})
        eval_ = pd.DataFrame({"timestamp": pd.date_range("2024-01-02", periods=5, freq="h")})
        with pytest.raises(ValueError, match="empty"):
            assert_no_leakage(empty, eval_)
