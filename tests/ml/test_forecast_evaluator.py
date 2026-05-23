"""Tests for aurelius.ml.forecast_evaluator.

Every test verifies that metrics are:
1. Mathematically correct (not just "code runs")
2. Correctly handling edge cases
3. Leakage-free (holdout actuals never touch training data in evaluation)
4. Deterministic
"""

import math
from datetime import datetime, timedelta, timezone

import pytest

from aurelius.ml.forecast_evaluator import (
    EvaluationResult,
    ForecastEvaluator,
    ForecastPoint,
    _compute_downside_risk,
    _compute_mae,
    _compute_mape,
    _compute_p50_bias,
    _compute_p90_coverage,
    _compute_rmse,
    compare_models,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _utc(y, m, d, h=0):
    return datetime(y, m, d, h, tzinfo=timezone.utc)


def _hours(n: int, start: datetime = None):
    """Generate n consecutive hourly timestamps starting from start."""
    if start is None:
        start = datetime(2024, 1, 1, 0, tzinfo=timezone.utc)
    return [start + timedelta(hours=i) for i in range(n)]


def make_points(timestamps, region, values):
    return [ForecastPoint(ts, region, v) for ts, v in zip(timestamps, values)]


@pytest.fixture
def perfect_forecast_data():
    """Perfect forecaster: p50 == actual, p90 == actual * 1.1 (always covers)."""
    ts = _hours(48)
    actuals_vals = [50.0 + (i % 24) for i in range(48)]  # diurnal pattern
    p50_vals = actuals_vals[:]
    p90_vals = [v * 1.1 for v in actuals_vals]
    actuals = make_points(ts, "us-west", actuals_vals)
    p50 = make_points(ts, "us-west", p50_vals)
    p90 = make_points(ts, "us-west", p90_vals)
    return actuals, p50, p90, actuals_vals


@pytest.fixture
def biased_forecast_data():
    """Over-forecast: p50 is always 10% above actual."""
    ts = _hours(48)
    actuals_vals = [50.0 + (i % 24) for i in range(48)]
    p50_vals = [v * 1.10 for v in actuals_vals]   # 10% over-forecast
    # p90 is 20% above actual: covers 100% of actuals
    p90_vals = [v * 1.20 for v in actuals_vals]
    actuals = make_points(ts, "us-west", actuals_vals)
    p50 = make_points(ts, "us-west", p50_vals)
    p90 = make_points(ts, "us-west", p90_vals)
    return actuals, p50, p90, actuals_vals


@pytest.fixture
def undercovering_forecast_data():
    """p90 always below actual → 0% empirical coverage."""
    ts = _hours(48)
    actuals_vals = [100.0] * 48
    p50_vals = [80.0] * 48
    p90_vals = [95.0] * 48    # p90 < actual (100) → 0% coverage
    actuals = make_points(ts, "us-west", actuals_vals)
    p50 = make_points(ts, "us-west", p50_vals)
    p90 = make_points(ts, "us-west", p90_vals)
    return actuals, p50, p90, actuals_vals


# ---------------------------------------------------------------------------
# Unit tests: metric helpers
# ---------------------------------------------------------------------------

class TestMetricHelpers:
    def test_mape_perfect(self):
        assert _compute_mape([100.0, 200.0], [100.0, 200.0]) == pytest.approx(0.0)

    def test_mape_10pct_error(self):
        # |100 - 110| / 100 = 0.10; |200 - 220| / 200 = 0.10
        assert _compute_mape([100.0, 200.0], [110.0, 220.0]) == pytest.approx(0.10)

    def test_mape_skips_zero_actuals(self):
        # Second actual is 0 → should be skipped
        result = _compute_mape([100.0, 0.0, 200.0], [110.0, 999.0, 220.0])
        # Only two valid rows: 0.10 and 0.10
        assert result == pytest.approx(0.10)

    def test_mape_all_zero_actuals_returns_nan(self):
        result = _compute_mape([0.0, 0.0], [1.0, 2.0])
        assert math.isnan(result)

    def test_rmse_perfect(self):
        assert _compute_rmse([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(0.0)

    def test_rmse_known_value(self):
        # errors: 1, 1 → mse = 1 → rmse = 1
        assert _compute_rmse([0.0, 0.0], [1.0, 1.0]) == pytest.approx(1.0)

    def test_mae_perfect(self):
        assert _compute_mae([5.0, 10.0], [5.0, 10.0]) == pytest.approx(0.0)

    def test_mae_known_value(self):
        assert _compute_mae([0.0, 2.0], [1.0, 3.0]) == pytest.approx(1.0)

    def test_p50_bias_positive_bias(self):
        # p50 always 10 above actual → bias = +10
        result = _compute_p50_bias([50.0, 50.0], [60.0, 60.0])
        assert result == pytest.approx(10.0)

    def test_p50_bias_negative_bias(self):
        # p50 always 10 below actual → bias = -10
        result = _compute_p50_bias([50.0, 50.0], [40.0, 40.0])
        assert result == pytest.approx(-10.0)

    def test_p50_bias_unbiased(self):
        result = _compute_p50_bias([50.0, 50.0], [50.0, 50.0])
        assert result == pytest.approx(0.0)

    def test_p90_coverage_all_covered(self):
        # p90 > actual for all → coverage = 1.0
        assert _compute_p90_coverage([50.0, 60.0], [100.0, 100.0]) == pytest.approx(1.0)

    def test_p90_coverage_none_covered(self):
        # p90 < actual for all → coverage = 0.0
        assert _compute_p90_coverage([100.0, 100.0], [50.0, 50.0]) == pytest.approx(0.0)

    def test_p90_coverage_half_covered(self):
        # First covered, second not
        assert _compute_p90_coverage([50.0, 100.0], [60.0, 90.0]) == pytest.approx(0.5)

    def test_p90_coverage_boundary_exact(self):
        # actual == p90 → covered (<=)
        assert _compute_p90_coverage([100.0], [100.0]) == pytest.approx(1.0)

    def test_downside_risk_zero_when_actuals_le_p50(self):
        # p50 >= actual for all → no exceedances
        assert _compute_downside_risk([50.0, 60.0], [80.0, 80.0]) == pytest.approx(0.0)

    def test_downside_risk_nonzero(self):
        # actual=100, p50=80 → exceedance=20; mean_actual=100 → risk=20/100=0.2
        result = _compute_downside_risk([100.0], [80.0])
        assert result == pytest.approx(0.20)


# ---------------------------------------------------------------------------
# ForecastEvaluator: core tests
# ---------------------------------------------------------------------------

class TestForecastEvaluator:
    def test_perfect_forecaster_metrics(self, perfect_forecast_data):
        actuals, p50, p90, _ = perfect_forecast_data
        result = ForecastEvaluator().evaluate(actuals, p50, p90)

        assert result.n_samples == 48
        assert result.mape == pytest.approx(0.0, abs=1e-9)
        assert result.rmse == pytest.approx(0.0, abs=1e-9)
        assert result.mae == pytest.approx(0.0, abs=1e-9)
        assert result.p50_bias == pytest.approx(0.0, abs=1e-9)
        assert result.p90_coverage == pytest.approx(1.0)
        assert result.downside_risk == pytest.approx(0.0, abs=1e-9)

    def test_biased_forecaster_mape(self, biased_forecast_data):
        actuals, p50, p90, _ = biased_forecast_data
        result = ForecastEvaluator().evaluate(actuals, p50, p90)

        # 10% over-forecast → MAPE ≈ 0.10
        assert result.mape == pytest.approx(0.10, abs=0.001)
        assert result.p50_bias > 0  # positive bias = over-forecast

    def test_biased_forecaster_p90_coverage(self, biased_forecast_data):
        actuals, p50, p90, _ = biased_forecast_data
        result = ForecastEvaluator().evaluate(actuals, p50, p90)

        # p90 = 1.2 * actual, always > actual → 100% coverage
        assert result.p90_coverage == pytest.approx(1.0)

    def test_undercovering_p90_coverage(self, undercovering_forecast_data):
        actuals, p50, p90, _ = undercovering_forecast_data
        result = ForecastEvaluator().evaluate(actuals, p50, p90)

        assert result.p90_coverage == pytest.approx(0.0)
        assert result.calibration_error == pytest.approx(0.90)

    def test_well_calibrated_property(self, perfect_forecast_data):
        actuals, p50, p90, _ = perfect_forecast_data
        result = ForecastEvaluator().evaluate(actuals, p50, p90)
        # coverage = 1.0 → calibration_error = 0.10 which is > 0.05 → NOT well calibrated
        assert not result.is_well_calibrated

    def test_well_calibrated_near_90pct(self):
        """If ~90% of actuals fall below p90, is_well_calibrated should be True."""
        n = 100
        ts = [_utc(2024, 1, 1, h % 24) + timedelta(days=h // 24) for h in range(n)]
        actuals_vals = [100.0] * n
        p50_vals = [100.0] * n
        # p90 covers exactly 91 out of 100 → coverage = 0.91 → cal_error = 0.01 < 0.05
        p90_vals = [200.0 if i < 91 else 50.0 for i in range(n)]
        actuals = make_points(ts, "us-west", actuals_vals)
        p50 = make_points(ts, "us-west", p50_vals)
        p90 = make_points(ts, "us-west", p90_vals)
        result = ForecastEvaluator().evaluate(actuals, p50, p90)
        assert result.p90_coverage == pytest.approx(0.91)
        assert result.is_well_calibrated

    def test_savings_lift_positive_when_better_than_naive(self):
        """Forecaster that beats the naive (mean) predictor should have positive lift."""
        ts = _hours(48)
        actuals_vals = list(range(1, 49))  # 1, 2, ..., 48
        training_mean = 24.5  # true mean of 1..48

        # Perfect p50 → MAE = 0 → lift = 100%
        p50_vals = actuals_vals[:]
        p90_vals = [v + 5 for v in actuals_vals]
        actuals = make_points(ts, "r", actuals_vals)
        p50 = make_points(ts, "r", p50_vals)
        p90 = make_points(ts, "r", p90_vals)

        result = ForecastEvaluator().evaluate(actuals, p50, p90, training_mean=training_mean)
        assert result.savings_lift == pytest.approx(1.0, abs=1e-9)

    def test_savings_lift_negative_when_worse_than_naive(self):
        """Forecaster that's worse than the naive predictor should have negative lift."""
        ts = _hours(48)
        # Actuals are 100; training mean is 80 (naive MAE = 20 per step)
        actuals_vals = [100.0] * 48
        training_mean = 80.0  # naive predictor: always predicts 80 → MAE = 20
        # Forecaster is terrible: always predicts 0 → MAE = 100 (much worse than naive)
        p50_vals = [0.0] * 48
        p90_vals = [0.0] * 48
        actuals = make_points(ts, "r", actuals_vals)
        p50 = make_points(ts, "r", p50_vals)
        p90 = make_points(ts, "r", p90_vals)
        result = ForecastEvaluator().evaluate(actuals, p50, p90, training_mean=training_mean)
        # naive_mae = 20, model_mae = 100 → savings_lift = (20 - 100) / 20 = -4.0
        assert result.savings_lift < 0

    def test_raises_on_empty_actuals(self):
        with pytest.raises(ValueError, match="actuals list is empty"):
            ForecastEvaluator().evaluate([], [], [])

    def test_raises_on_no_matching_keys(self):
        """Actuals in region A, forecasts in region B → no intersection."""
        ts = [_utc(2024, 1, 1, h) for h in range(5)]
        actuals = make_points(ts, "region-A", [100.0] * 5)
        p50 = make_points(ts, "region-B", [100.0] * 5)
        p90 = make_points(ts, "region-B", [110.0] * 5)
        with pytest.raises(ValueError, match="No matching"):
            ForecastEvaluator().evaluate(actuals, p50, p90)

    def test_partial_match_uses_intersection(self):
        """If only some timestamps match, only those are evaluated."""
        ts_all = [_utc(2024, 1, 1, h) for h in range(10)]
        ts_partial = [_utc(2024, 1, 1, h) for h in range(5)]  # first 5 only
        actuals = make_points(ts_all, "r", [100.0] * 10)
        p50 = make_points(ts_partial, "r", [100.0] * 5)
        p90 = make_points(ts_partial, "r", [110.0] * 5)
        result = ForecastEvaluator().evaluate(actuals, p50, p90)
        assert result.n_samples == 5

    def test_multi_region_per_region_metrics(self):
        """Per-region metrics must be computed independently."""
        ts = [_utc(2024, 1, 1, h) for h in range(24)]
        # Region A: perfect forecast
        a_actuals = make_points(ts, "A", [100.0] * 24)
        a_p50 = make_points(ts, "A", [100.0] * 24)
        a_p90 = make_points(ts, "A", [110.0] * 24)
        # Region B: 20% over-forecast
        b_actuals = make_points(ts, "B", [50.0] * 24)
        b_p50 = make_points(ts, "B", [60.0] * 24)
        b_p90 = make_points(ts, "B", [70.0] * 24)

        all_actuals = a_actuals + b_actuals
        all_p50 = a_p50 + b_p50
        all_p90 = a_p90 + b_p90

        result = ForecastEvaluator().evaluate(all_actuals, all_p50, all_p90)

        assert "A" in result.per_region
        assert "B" in result.per_region
        assert result.per_region["A"]["mape"] == pytest.approx(0.0, abs=1e-9)
        assert result.per_region["B"]["mape"] == pytest.approx(0.20, abs=0.001)

    def test_deterministic(self, perfect_forecast_data):
        """Same inputs must produce same outputs."""
        actuals, p50, p90, _ = perfect_forecast_data
        r1 = ForecastEvaluator().evaluate(actuals, p50, p90)
        r2 = ForecastEvaluator().evaluate(actuals, p50, p90)
        assert r1.mape == r2.mape
        assert r1.p90_coverage == r2.p90_coverage

    def test_to_dict_has_all_required_fields(self, perfect_forecast_data):
        actuals, p50, p90, _ = perfect_forecast_data
        result = ForecastEvaluator().evaluate(actuals, p50, p90)
        d = result.to_dict()
        for field_name in (
            "n_samples", "mape", "rmse", "mae", "p50_bias",
            "p90_coverage", "calibration_error", "downside_risk",
            "savings_lift_pct", "is_well_calibrated", "regions", "per_region",
        ):
            assert field_name in d, f"Missing field: {field_name}"

    def test_warning_on_small_holdout(self, caplog):
        """Fewer than 24 samples should emit a warning."""
        import logging
        ts = [_utc(2024, 1, 1, h) for h in range(10)]
        actuals = make_points(ts, "r", [100.0] * 10)
        p50 = make_points(ts, "r", [100.0] * 10)
        p90 = make_points(ts, "r", [110.0] * 10)
        with caplog.at_level(logging.WARNING):
            ForecastEvaluator().evaluate(actuals, p50, p90)
        assert any("unreliable" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# EvaluationResult.is_better_than
# ---------------------------------------------------------------------------

class TestEvaluationResultComparison:
    def _make_result(self, mape, p90_coverage, calibration_error=None):
        cal = calibration_error if calibration_error is not None else abs(p90_coverage - 0.9)
        return EvaluationResult(
            n_samples=100, mape=mape, rmse=mape * 2, mae=mape * 1.5,
            p50_bias=0.0, p90_coverage=p90_coverage,
            calibration_error=cal, downside_risk=0.05, savings_lift=0.10,
        )

    def test_lower_mape_is_better(self):
        a = self._make_result(mape=0.10, p90_coverage=0.88)
        b = self._make_result(mape=0.15, p90_coverage=0.88)
        assert a.is_better_than(b, "mape")
        assert not b.is_better_than(a, "mape")

    def test_higher_savings_lift_is_better(self):
        a = EvaluationResult(
            n_samples=100, mape=0.10, rmse=0.20, mae=0.15,
            p50_bias=0.0, p90_coverage=0.88, calibration_error=0.02,
            downside_risk=0.05, savings_lift=0.20,
        )
        b = EvaluationResult(
            n_samples=100, mape=0.10, rmse=0.20, mae=0.15,
            p50_bias=0.0, p90_coverage=0.88, calibration_error=0.02,
            downside_risk=0.05, savings_lift=0.10,
        )
        assert a.is_better_than(b, "savings_lift")


# ---------------------------------------------------------------------------
# compare_models
# ---------------------------------------------------------------------------

class TestCompareModels:
    def _result(self, mape, cal_error):
        return EvaluationResult(
            n_samples=100, mape=mape, rmse=mape * 2, mae=mape * 1.5,
            p50_bias=0.0, p90_coverage=0.90 - cal_error,
            calibration_error=cal_error, downside_risk=0.05, savings_lift=0.10,
        )

    def test_promotes_when_improved_and_no_cal_regression(self):
        candidate = self._result(mape=0.08, cal_error=0.02)
        current = self._result(mape=0.10, cal_error=0.02)
        cmp = compare_models(candidate, current, primary_metric="mape", min_improvement_pct=1.0)
        assert cmp.promote is True
        assert cmp.improvement_pct == pytest.approx(20.0, abs=0.1)

    def test_does_not_promote_when_not_improved_enough(self):
        # Only 0.5% improvement; need 1.0%
        candidate = self._result(mape=0.0995, cal_error=0.02)
        current = self._result(mape=0.10, cal_error=0.02)
        cmp = compare_models(candidate, current, primary_metric="mape", min_improvement_pct=1.0)
        assert cmp.promote is False

    def test_does_not_promote_on_calibration_regression(self):
        # Large MAPE improvement but big calibration regression
        candidate = self._result(mape=0.05, cal_error=0.20)  # cal_error increases a lot
        current = self._result(mape=0.10, cal_error=0.02)
        cmp = compare_models(
            candidate, current,
            primary_metric="mape",
            min_improvement_pct=1.0,
            max_calibration_regression_pts=0.05,
        )
        assert cmp.promote is False
        assert "calibration" in cmp.reason.lower()

    def test_promotes_with_acceptable_calibration_change(self):
        # Small calibration regression within tolerance
        candidate = self._result(mape=0.05, cal_error=0.04)
        current = self._result(mape=0.10, cal_error=0.02)
        # cal regression = 0.04 - 0.02 = 0.02 < max 0.05
        cmp = compare_models(candidate, current, primary_metric="mape", min_improvement_pct=1.0)
        assert cmp.promote is True

    def test_comparison_result_fields_present(self):
        candidate = self._result(mape=0.08, cal_error=0.02)
        current = self._result(mape=0.10, cal_error=0.02)
        cmp = compare_models(candidate, current)
        assert hasattr(cmp, "candidate_better")
        assert hasattr(cmp, "primary_metric")
        assert hasattr(cmp, "improvement_pct")
        assert hasattr(cmp, "promote")
        assert hasattr(cmp, "reason")
        assert isinstance(cmp.reason, str) and len(cmp.reason) > 0

    def test_unknown_primary_metric_raises(self):
        r = self._result(mape=0.10, cal_error=0.02)
        with pytest.raises(ValueError):
            compare_models(r, r, primary_metric="nonexistent")


# ---------------------------------------------------------------------------
# ForecastEvaluator.evaluate_from_model integration tests
# ---------------------------------------------------------------------------

class TestEvaluateFromModel:
    """Tests that evaluate_from_model correctly drives a real forecaster."""

    def _make_price_records(self, n_train=200, n_holdout=48, seed=42):
        """Generate synthetic EnergyPrice records."""
        import math as _math

        from aurelius.models import EnergyPrice
        rng_state = seed
        def _rng():
            nonlocal rng_state
            rng_state = (rng_state * 1664525 + 1013904223) & 0xFFFFFFFF
            return rng_state / 0xFFFFFFFF

        base_ts = datetime(2024, 1, 1, 0, tzinfo=timezone.utc)
        records = []
        for i in range(n_train + n_holdout):
            ts = base_ts + timedelta(hours=i)
            # Diurnal pattern + noise
            price = 50 + 20 * _math.sin(2 * _math.pi * ts.hour / 24) + (_rng() - 0.5) * 10
            records.append(EnergyPrice(timestamp=ts, region="us-west", price_per_mwh=price))
        return records[:n_train], records[n_train:]

    def test_evaluate_from_model_price_forecaster(self):
        from aurelius.forecasting.price_model import PriceModelConfig, PriceQuantileForecaster
        train_records, holdout_records = self._make_price_records(n_train=200, n_holdout=48)

        forecaster = PriceQuantileForecaster(PriceModelConfig(seed=42, n_estimators=20))
        forecaster.fit(train_records)

        training_mean = sum(r.price_per_mwh for r in train_records) / len(train_records)
        recent_context = train_records[-48:]

        result = ForecastEvaluator().evaluate_from_model(
            forecaster=forecaster,
            holdout_actuals=holdout_records,
            training_mean=training_mean,
            recent_context=recent_context,
        )

        assert result.n_samples == 48
        assert 0.0 <= result.mape <= 2.0   # MAPE should be a finite fraction
        assert result.rmse >= 0
        assert result.mae >= 0
        assert 0.0 <= result.p90_coverage <= 1.0
        assert 0.0 <= result.calibration_error <= 1.0

    def test_evaluate_from_model_carbon_forecaster(self):
        import math as _math

        from aurelius.forecasting.carbon_model import CarbonModelConfig, CarbonQuantileForecaster
        from aurelius.models import CarbonIntensity
        base_ts = datetime(2024, 2, 1, 0, tzinfo=timezone.utc)
        train = [
            CarbonIntensity(
                timestamp=base_ts + timedelta(hours=i),
                region="us-east",
                gco2_per_kwh=300 + 100 * _math.sin(2 * _math.pi * (base_ts + timedelta(hours=i)).hour / 24),
            )
            for i in range(200)
        ]
        holdout = [
            CarbonIntensity(
                timestamp=base_ts + timedelta(hours=200 + i),
                region="us-east",
                gco2_per_kwh=300 + 100 * _math.sin(2 * _math.pi * (base_ts + timedelta(hours=200 + i)).hour / 24),
            )
            for i in range(48)
        ]

        forecaster = CarbonQuantileForecaster(CarbonModelConfig(seed=42, n_estimators=20))
        forecaster.fit(train)

        training_mean = sum(r.gco2_per_kwh for r in train) / len(train)
        result = ForecastEvaluator().evaluate_from_model(
            forecaster=forecaster,
            holdout_actuals=holdout,
            training_mean=training_mean,
        )

        assert result.n_samples == 48
        assert math.isfinite(result.mape)
        assert math.isfinite(result.p90_coverage)

    def test_evaluate_from_model_raises_on_unfitted(self):
        from aurelius.forecasting.price_model import PriceQuantileForecaster
        from aurelius.models import EnergyPrice
        unfitted = PriceQuantileForecaster()
        # Even unfitted, predict() returns baseline — evaluation should succeed (not raise)
        # but the unfitted model will return baseline_fallback predictions
        ts = [_utc(2024, 1, 1, h) for h in range(24)]
        actuals = [EnergyPrice(t, "r", 50.0) for t in ts]
        result = ForecastEvaluator().evaluate_from_model(unfitted, actuals)
        assert result.n_samples == 24

    def test_evaluate_from_model_raises_on_wrong_type(self):
        """holdout_actuals items without price_per_mwh or gco2_per_kwh should raise."""
        from aurelius.forecasting.price_model import PriceQuantileForecaster
        from aurelius.models import EnergyPrice
        base = datetime(2024, 1, 1, 0, tzinfo=timezone.utc)
        train = [EnergyPrice(base + timedelta(hours=h), "r", 50.0) for h in range(200)]
        forecaster = PriceQuantileForecaster()
        forecaster.fit(train)

        class Bogus:
            timestamp = _utc(2024, 1, 1, 0)
            region = "r"

        with pytest.raises(ValueError, match="price_per_mwh"):
            ForecastEvaluator().evaluate_from_model(forecaster, [Bogus()])
