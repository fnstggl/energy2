"""Tests for QuantileSafetyGate – fail-closed behaviour."""

import pytest
from dataclasses import dataclass
from typing import Optional

from aurelius.safety.quantile_gate import QuantileSafetyGate, QuantileGateConfig


@dataclass
class MockDecision:
    job_id: str
    forecast: Optional[dict] = None


GATE = QuantileSafetyGate()


def _config(**kwargs) -> QuantileGateConfig:
    defaults = dict(
        enabled=True,
        min_expected_savings_pct=5.0,
        max_downside_risk_pct=10.0,
        metric="energy_cost",
    )
    defaults.update(kwargs)
    return QuantileGateConfig(**defaults)


def _good_forecast():
    return {"energy_cost": {"p50": 90.0, "p90": 105.0, "baseline": 100.0}}


def _bad_forecast():
    # p50 worse than baseline → fails expected savings
    return {"energy_cost": {"p50": 110.0, "p90": 120.0, "baseline": 100.0}}


# ---------------------------------------------------------------------------
# Fail-closed: missing forecast BLOCKS the decision
# ---------------------------------------------------------------------------

class TestFailClosed:
    def test_none_forecast_is_blocked(self):
        decisions = [MockDecision(job_id="j1", forecast=None)]
        result = GATE.filter(decisions, _config())
        assert len(result) == 0, "Missing forecast must be BLOCKED (fail-closed)"

    def test_missing_metric_key_is_blocked(self):
        decisions = [MockDecision(job_id="j1", forecast={"carbon": {"p50": 10, "p90": 15, "baseline": 20}})]
        result = GATE.filter(decisions, _config(metric="energy_cost"))
        assert len(result) == 0

    def test_missing_p50_is_blocked(self):
        decisions = [MockDecision(
            job_id="j1",
            forecast={"energy_cost": {"p90": 105.0, "baseline": 100.0}},  # no p50
        )]
        result = GATE.filter(decisions, _config())
        assert len(result) == 0

    def test_missing_p90_is_blocked(self):
        decisions = [MockDecision(
            job_id="j1",
            forecast={"energy_cost": {"p50": 90.0, "baseline": 100.0}},  # no p90
        )]
        result = GATE.filter(decisions, _config())
        assert len(result) == 0

    def test_zero_baseline_is_blocked(self):
        decisions = [MockDecision(
            job_id="j1",
            forecast={"energy_cost": {"p50": 90.0, "p90": 105.0, "baseline": 0.0}},
        )]
        result = GATE.filter(decisions, _config())
        assert len(result) == 0

    def test_negative_baseline_is_blocked(self):
        decisions = [MockDecision(
            job_id="j1",
            forecast={"energy_cost": {"p50": 90.0, "p90": 105.0, "baseline": -5.0}},
        )]
        result = GATE.filter(decisions, _config())
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Gate disabled passes all (including missing forecasts)
# ---------------------------------------------------------------------------

class TestGateDisabled:
    def test_disabled_passes_all_even_without_forecast(self):
        decisions = [
            MockDecision(job_id="j1", forecast=None),
            MockDecision(job_id="j2", forecast=None),
        ]
        result = GATE.filter(decisions, QuantileGateConfig(enabled=False))
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Valid forecast evaluation
# ---------------------------------------------------------------------------

class TestValidForecast:
    def test_good_forecast_passes(self):
        decisions = [MockDecision(job_id="j1", forecast=_good_forecast())]
        result = GATE.filter(decisions, _config())
        assert len(result) == 1

    def test_bad_forecast_blocked(self):
        decisions = [MockDecision(job_id="j1", forecast=_bad_forecast())]
        result = GATE.filter(decisions, _config())
        assert len(result) == 0

    def test_downside_risk_too_high(self):
        forecast = {"energy_cost": {"p50": 90.0, "p90": 130.0, "baseline": 100.0}}
        decisions = [MockDecision(job_id="j1", forecast=forecast)]
        result = GATE.filter(decisions, _config(max_downside_risk_pct=5.0))
        assert len(result) == 0

    def test_order_preserved(self):
        decisions = [
            MockDecision(job_id="j1", forecast=_good_forecast()),
            MockDecision(job_id="j2", forecast=_bad_forecast()),
            MockDecision(job_id="j3", forecast=_good_forecast()),
        ]
        result = GATE.filter(decisions, _config())
        assert [d.job_id for d in result] == ["j1", "j3"]

    def test_both_metrics_required(self):
        forecast = {
            "energy_cost": {"p50": 90.0, "p90": 105.0, "baseline": 100.0},
            "carbon": {"p50": 500.0, "p90": 600.0, "baseline": 100.0},  # terrible carbon
        }
        decisions = [MockDecision(job_id="j1", forecast=forecast)]
        result = GATE.filter(decisions, _config(metric="both"))
        assert len(result) == 0

    def test_determinism(self):
        decisions = [
            MockDecision(job_id="j1", forecast=_good_forecast()),
            MockDecision(job_id="j2", forecast=None),
        ]
        r1 = GATE.filter(decisions, _config())
        r2 = GATE.filter(decisions, _config())
        assert [d.job_id for d in r1] == [d.job_id for d in r2]
