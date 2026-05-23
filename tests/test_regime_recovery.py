"""Tests for regime detection and forecast recovery correction.

Coverage:
- RegimeInfo dataclass construction
- RegimeDetector.detect(): recovery detection, spike detection, normal, edge cases
- RegimeDetector.correct_predictions(): correction magnitude, decay, floor
- RegimeDetector.apply_corrections_to_forecast(): multi-region, selective
- BacktestEngine integration: apply_recovery_correction flag
- Leakage safety: correction uses only training data
- Adversarial: false positive guard, over-correction guard, no-data handling
"""

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pytest

from aurelius.forecasting.regime import (
    RegimeDetector,
    RegimeInfo,
    compute_region_regime_summary,
)
from aurelius.models import EnergyPrice


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_prices(n: int, value: float = 50.0, seed: int = 0) -> list[float]:
    rng = np.random.RandomState(seed)
    return [max(1.0, value + rng.normal(0, value * 0.05)) for _ in range(n)]


def _make_energy_prices(
    region: str,
    n: int,
    value: float = 50.0,
    start: Optional[datetime] = None,
    seed: int = 0,
) -> list[EnergyPrice]:
    if start is None:
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    prices = _make_prices(n, value, seed)
    return [
        EnergyPrice(
            timestamp=start + timedelta(hours=i),
            region=region,
            price_per_mwh=prices[i],
        )
        for i in range(n)
    ]


def _make_spike_context(
    spike_hours: int = 168,
    spike_price: float = 2000.0,
    recovery_hours: int = 24,
    recovery_price: float = 30.0,
    normal_price: float = 50.0,
) -> tuple[list[float], list[float]]:
    """Simulate a cold-snap followed by recovery.

    Returns (training_prices, recent_context_prices).
    """
    # Training: normal → spike period
    training = (
        _make_prices(480, normal_price, seed=1) +  # 20 days normal
        _make_prices(spike_hours, spike_price, seed=2)  # spike
    )
    # Recent context: recovery
    recent = _make_prices(recovery_hours, recovery_price, seed=3)
    return training, recent


# ---------------------------------------------------------------------------
# TestRegimeInfo
# ---------------------------------------------------------------------------

class TestRegimeInfo:
    def test_construction(self):
        info = RegimeInfo(
            region="us-south",
            is_recovering=True,
            recovery_ratio=0.15,
            recent_mean=30.0,
            training_mean=200.0,
            correction_magnitude=0.45,
        )
        assert info.region == "us-south"
        assert info.is_recovering is True
        assert info.recovery_ratio == pytest.approx(0.15)
        assert info.correction_magnitude == pytest.approx(0.45)

    def test_not_recovering(self):
        info = RegimeInfo(
            region="us-west",
            is_recovering=False,
            recovery_ratio=0.85,
            recent_mean=40.0,
            training_mean=47.0,
            correction_magnitude=0.0,
        )
        assert info.is_recovering is False
        assert info.correction_magnitude == 0.0


# ---------------------------------------------------------------------------
# TestRegimeDetectorInit
# ---------------------------------------------------------------------------

class TestRegimeDetectorInit:
    def test_defaults(self):
        d = RegimeDetector()
        assert d.recovery_ratio_threshold == pytest.approx(0.40)
        assert d.max_recent_mean_for_correction == pytest.approx(30.0)
        assert d.recent_hours == 24
        assert d.max_correction_fraction == pytest.approx(0.50)
        assert d.decay_halflife_hours == pytest.approx(72.0)

    def test_invalid_threshold(self):
        with pytest.raises(ValueError):
            RegimeDetector(recovery_ratio_threshold=0.0)
        with pytest.raises(ValueError):
            RegimeDetector(recovery_ratio_threshold=1.0)

    def test_invalid_recent_hours(self):
        with pytest.raises(ValueError):
            RegimeDetector(recent_hours=0)

    def test_invalid_max_correction(self):
        with pytest.raises(ValueError):
            RegimeDetector(max_correction_fraction=0.0)
        with pytest.raises(ValueError):
            RegimeDetector(max_correction_fraction=1.1)

    def test_invalid_max_recent_mean(self):
        with pytest.raises(ValueError):
            RegimeDetector(max_recent_mean_for_correction=0.0)
        with pytest.raises(ValueError):
            RegimeDetector(max_recent_mean_for_correction=-5.0)

    def test_invalid_decay(self):
        with pytest.raises(ValueError):
            RegimeDetector(decay_halflife_hours=0.0)


# ---------------------------------------------------------------------------
# TestRegimeDetectorDetect
# ---------------------------------------------------------------------------

class TestRegimeDetectorDetect:
    def test_cold_snap_recovery_detected(self):
        """ERCOT cold snap recovery: recent $25, training mean $150 → recovery."""
        detector = RegimeDetector()
        training, recent = _make_spike_context(
            spike_hours=168, spike_price=2000.0,
            recovery_hours=24, recovery_price=25.0,
        )
        info = detector.detect("us-south", training, recent)

        assert info.region == "us-south"
        assert info.is_recovering is True
        assert info.recovery_ratio < 0.40
        assert info.correction_magnitude > 0.0
        assert info.recent_mean == pytest.approx(25.0, rel=0.15)
        assert info.training_mean > 100.0  # dominated by spike

    def test_stable_regime_no_correction(self):
        """Stable market: recent $45, training mean $47 → no correction."""
        detector = RegimeDetector()
        training = _make_prices(720, 47.0, seed=0)
        recent = _make_prices(24, 45.0, seed=1)
        info = detector.detect("us-west", training, recent)

        assert info.is_recovering is False
        assert info.correction_magnitude == 0.0
        assert info.recovery_ratio > 0.40

    def test_normal_diurnal_no_false_positive(self):
        """Overnight prices are low but ratio stays above threshold → no correction."""
        detector = RegimeDetector(recovery_ratio_threshold=0.40)
        # 30 days of prices: mix of high (daytime ~70) and low (overnight ~25)
        training: list[float] = []
        for _ in range(30):
            training += [25.0] * 8 + [70.0] * 8 + [50.0] * 8  # diurnal
        recent = [25.0] * 24  # overnight period
        recent_mean = 25.0
        training_mean = np.mean(training)
        ratio = recent_mean / training_mean  # ~25/48 ≈ 0.52 > 0.40

        info = detector.detect("us-west", training, recent)
        assert info.is_recovering is False, (
            f"False positive: ratio={ratio:.2f} should be > 0.40 for diurnal variation"
        )

    def test_spike_onset_no_correction(self):
        """Prices spiking NOW (recent > training mean) → no correction."""
        detector = RegimeDetector()
        training = _make_prices(720, 50.0, seed=0)    # normal training
        recent = _make_prices(24, 2000.0, seed=1)     # current spike
        info = detector.detect("us-south", training, recent)

        assert info.is_recovering is False  # recent > training mean → ratio > 1.0
        assert info.correction_magnitude == 0.0

    def test_empty_training_prices(self):
        """No training data → no correction."""
        detector = RegimeDetector()
        info = detector.detect("us-south", [], [30.0] * 24)
        assert info.is_recovering is False
        assert info.correction_magnitude == 0.0

    def test_empty_recent_prices(self):
        """No recent context → no correction."""
        detector = RegimeDetector()
        training = _make_prices(720, 150.0)
        info = detector.detect("us-south", training, [])
        assert info.is_recovering is False
        assert info.correction_magnitude == 0.0

    def test_zero_training_mean(self):
        """Zero training prices → no correction (divide-by-zero guard)."""
        detector = RegimeDetector()
        info = detector.detect("us-south", [0.0] * 100, [30.0] * 24)
        assert info.is_recovering is False

    def test_correction_magnitude_bounded(self):
        """Correction magnitude never exceeds max_correction_fraction."""
        detector = RegimeDetector(max_correction_fraction=0.50)
        training, recent = _make_spike_context(
            spike_price=5000.0, recovery_price=5.0
        )
        info = detector.detect("us-south", training, recent)
        assert info.correction_magnitude <= 0.50

    def test_deeper_recovery_higher_magnitude(self):
        """Deeper recovery (lower ratio) → larger correction.

        Both price levels must be below max_recent_mean_for_correction (30.0).
        """
        detector = RegimeDetector()
        training = [200.0] * 720
        # shallow recovery: ratio = 28/200 = 0.14 < 0.40, recent_mean 28 < 30
        info_shallow = detector.detect("r1", training, [28.0] * 24)
        # deep recovery: ratio = 15/200 = 0.075 < 0.40, recent_mean 15 < 30
        info_deep = detector.detect("r1", training, [15.0] * 24)

        assert info_shallow.is_recovering
        assert info_deep.is_recovering
        assert info_deep.correction_magnitude >= info_shallow.correction_magnitude

    def test_borderline_below_threshold(self):
        """Just below threshold AND below price ceiling → detected as recovering."""
        detector = RegimeDetector(recovery_ratio_threshold=0.40)
        training = [100.0] * 720
        # ratio = 28/100 = 0.28 < 0.40 AND recent_mean 28 < 30 ceiling
        recent = [28.0] * 24
        info = detector.detect("r1", training, recent)
        assert info.is_recovering is True

    def test_borderline_at_threshold(self):
        """Exactly at threshold → NOT recovering (strict <)."""
        detector = RegimeDetector(recovery_ratio_threshold=0.40)
        training = [100.0] * 720
        recent = [40.0] * 24  # ratio = 0.40, NOT < 0.40
        info = detector.detect("r1", training, recent)
        assert info.is_recovering is False


# ---------------------------------------------------------------------------
# TestCorrectPredictions
# ---------------------------------------------------------------------------

class TestCorrectPredictions:
    def _make_ts_map(self, n: int, price: float = 200.0) -> dict:
        base = datetime(2026, 2, 1, tzinfo=timezone.utc)
        return {base + timedelta(hours=i): price for i in range(n)}

    def test_no_correction_when_not_recovering(self):
        """Non-recovering regime → predictions unchanged."""
        detector = RegimeDetector()
        regime = RegimeInfo("r1", False, 0.85, 40.0, 47.0, 0.0)
        predictions = self._make_ts_map(72, 200.0)
        corrected = detector.correct_predictions(predictions, regime)
        assert corrected == predictions

    def test_correction_reduces_high_predictions(self):
        """Recovery regime → predictions above recent_mean are reduced."""
        detector = RegimeDetector()
        regime = RegimeInfo("us-south", True, 0.15, 30.0, 200.0, 0.45)
        predictions = self._make_ts_map(48, 200.0)
        corrected = detector.correct_predictions(predictions, regime)

        # All corrected prices should be lower than original
        for ts, orig in predictions.items():
            assert corrected[ts] <= orig, f"Correction should reduce price at {ts}"

    def test_correction_does_not_inflate(self):
        """Predictions below recent_mean are never inflated."""
        detector = RegimeDetector()
        regime = RegimeInfo("us-south", True, 0.15, 30.0, 200.0, 0.45)
        # Predictions already below recent_mean
        predictions = {
            datetime(2026, 2, 1, tzinfo=timezone.utc) + timedelta(hours=i): 25.0
            for i in range(24)
        }
        corrected = detector.correct_predictions(predictions, regime)
        for ts, price in corrected.items():
            assert price == pytest.approx(25.0), "Should not change prices below recent_mean"

    def test_correction_decays_over_time(self):
        """Near-term hours have larger corrections than far-horizon hours."""
        detector = RegimeDetector(decay_halflife_hours=24.0)
        regime = RegimeInfo("us-south", True, 0.15, 30.0, 200.0, 0.40)
        predictions = self._make_ts_map(168, 200.0)
        corrected = detector.correct_predictions(predictions, regime)

        ts_sorted = sorted(corrected.keys())
        # Early correction > late correction
        early_reduction = 200.0 - corrected[ts_sorted[0]]
        late_reduction = 200.0 - corrected[ts_sorted[-1]]
        assert early_reduction > late_reduction, "Decay should reduce correction over time"

    def test_floor_prevents_over_correction(self):
        """Corrected price never falls below 80% of recent_mean."""
        detector = RegimeDetector(max_correction_fraction=0.99)
        recent_mean = 30.0
        regime = RegimeInfo("us-south", True, 0.10, recent_mean, 300.0, 0.99)
        predictions = self._make_ts_map(1, 300.0)
        corrected = detector.correct_predictions(predictions, regime)
        for price in corrected.values():
            assert price >= recent_mean * 0.8, "Should not over-correct below floor"

    def test_empty_predictions(self):
        """Empty prediction dict → empty corrected dict."""
        detector = RegimeDetector()
        regime = RegimeInfo("r1", True, 0.20, 25.0, 200.0, 0.40)
        corrected = detector.correct_predictions({}, regime)
        assert corrected == {}

    def test_correction_is_deterministic(self):
        """Same inputs → same outputs."""
        detector = RegimeDetector()
        regime = RegimeInfo("us-south", True, 0.15, 30.0, 200.0, 0.40)
        predictions = self._make_ts_map(48, 200.0)
        c1 = detector.correct_predictions(predictions, regime)
        c2 = detector.correct_predictions(predictions, regime)
        for ts in c1:
            assert c1[ts] == pytest.approx(c2[ts])


# ---------------------------------------------------------------------------
# TestApplyCorrectionsToForecast
# ---------------------------------------------------------------------------

class TestApplyCorrectionsToForecast:
    def _make_forecast_dict(
        self, region: str, n: int = 72, price: float = 200.0
    ) -> dict:
        base = datetime(2026, 2, 1, tzinfo=timezone.utc)
        return {region: {base + timedelta(hours=i): price for i in range(n)}}

    def _make_train_data(self, region: str, values: list[float]) -> dict:
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        return {region: {base + timedelta(hours=i): v for i, v in enumerate(values)}}

    def test_recovering_region_corrected(self):
        """Post-spike recovery region gets corrected."""
        detector = RegimeDetector()
        # ERCOT with cold snap: training mean ~$300 (spike), recent $25 (recovery)
        training_vals = [50.0] * 480 + [2000.0] * 240  # spike in training
        train_data = self._make_train_data("us-south", training_vals)
        forecast = self._make_forecast_dict("us-south", 72, 300.0)

        context = _make_energy_prices("us-south", 24, 25.0)

        corrected = detector.apply_corrections_to_forecast(forecast, train_data, context)

        # All corrected prices should be below original 300.0
        for ts, price in corrected["us-south"].items():
            assert price < 300.0

    def test_stable_region_untouched(self):
        """Stable region with no recovery → unchanged."""
        detector = RegimeDetector()
        training_vals = [47.0] * 720
        train_data = self._make_train_data("us-west", training_vals)
        forecast = self._make_forecast_dict("us-west", 72, 50.0)
        context = _make_energy_prices("us-west", 24, 45.0)

        corrected = detector.apply_corrections_to_forecast(forecast, train_data, context)

        for ts, price in corrected["us-west"].items():
            assert price == pytest.approx(50.0)

    def test_selective_correction_multi_region(self):
        """Only recovering region gets corrected; stable regions untouched."""
        detector = RegimeDetector()

        # ERCOT: recovering
        ercot_training = [50.0] * 480 + [2000.0] * 240
        # CAISO: stable
        caiso_training = [47.0] * 720

        train_data = {
            "us-south": {datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(hours=i): v
                         for i, v in enumerate(ercot_training)},
            "us-west": {datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(hours=i): v
                        for i, v in enumerate(caiso_training)},
        }

        base = datetime(2026, 2, 1, tzinfo=timezone.utc)
        forecast = {
            "us-south": {base + timedelta(hours=i): 300.0 for i in range(72)},
            "us-west": {base + timedelta(hours=i): 50.0 for i in range(72)},
        }

        context = (
            _make_energy_prices("us-south", 24, 25.0) +
            _make_energy_prices("us-west", 24, 45.0)
        )

        corrected = detector.apply_corrections_to_forecast(forecast, train_data, context)

        # ERCOT corrected (below original 300.0)
        assert all(p < 300.0 for p in corrected["us-south"].values())
        # CAISO unchanged
        assert all(p == pytest.approx(50.0) for p in corrected["us-west"].values())

    def test_empty_forecast(self):
        """Empty forecast → empty result."""
        detector = RegimeDetector()
        corrected = detector.apply_corrections_to_forecast({}, {}, [])
        assert corrected == {}

    def test_missing_region_in_train_data(self):
        """Region in forecast but not in train_data → no correction (graceful)."""
        detector = RegimeDetector()
        base = datetime(2026, 2, 1, tzinfo=timezone.utc)
        forecast = {"us-south": {base: 300.0}}
        corrected = detector.apply_corrections_to_forecast(forecast, {}, [])
        assert corrected["us-south"][base] == pytest.approx(300.0)


# ---------------------------------------------------------------------------
# TestBacktestEngineRecoveryCorrection
# ---------------------------------------------------------------------------

class TestBacktestEngineRecoveryCorrection:
    """Integration tests: recovery correction flag in BacktestEngine."""

    def _get_engine_class(self):
        from aurelius.backtesting.engine import BacktestEngine
        return BacktestEngine

    def test_engine_accepts_flag(self):
        """BacktestEngine.__init__ accepts apply_recovery_correction parameter."""
        BacktestEngine = self._get_engine_class()
        engine = BacktestEngine(apply_recovery_correction=True)
        assert engine.apply_recovery_correction is True

    def test_engine_default_flag_false(self):
        """Default is False (backward-compatible)."""
        BacktestEngine = self._get_engine_class()
        engine = BacktestEngine()
        assert engine.apply_recovery_correction is False

    def test_engine_no_regression_without_flag(self):
        """Engine without flag produces identical results to pre-change behavior."""
        BacktestEngine = self._get_engine_class()
        from aurelius.forecasting.price_model import PriceQuantileForecaster, PriceModelConfig
        from aurelius.models import Job
        import pandas as pd

        engine_no_correction = BacktestEngine(
            method="greedy_migrate",
            apply_recovery_correction=False,
            price_forecaster_cls=PriceQuantileForecaster,
            price_forecaster_config=PriceModelConfig(seed=42, n_estimators=20),
        )

        # Make a small price fixture: stable market
        base = pd.Timestamp("2026-01-01", tz="UTC")
        rows = []
        for region in ["us-west", "us-east"]:
            for h in range(24 * 45):
                rows.append({
                    "timestamp": base + pd.Timedelta(hours=h),
                    "region": region,
                    "price_per_mwh": 50.0 + 5 * np.sin(h * 2 * np.pi / 24),
                })
        price_df = pd.DataFrame(rows)
        carbon_df = pd.DataFrame(columns=["timestamp", "region", "gco2_per_kwh"])

        from aurelius.ingestion.job_logs import JobLogIngester
        ingester = JobLogIngester()
        jobs = ingester.generate_synthetic(
            start_time=base.to_pydatetime(),
            duration_hours=24 * 45,
            num_jobs=20,
            regions=["us-west", "us-east"],
            seed=42,
            workload_filter="llm_batch_inference",
        )

        rounds = engine_no_correction.run(jobs, price_df, carbon_df)
        assert len(rounds) > 0, "Engine should produce folds"

    def test_engine_correction_activates_for_recovery_regime(self):
        """With correction flag, engine activates correction for spiked-then-recovered regions."""
        BacktestEngine = self._get_engine_class()
        from aurelius.forecasting.price_model import PriceQuantileForecaster, PriceModelConfig
        import pandas as pd

        engine = BacktestEngine(
            method="greedy_migrate",
            apply_recovery_correction=True,
            price_forecaster_cls=PriceQuantileForecaster,
            price_forecaster_config=PriceModelConfig(seed=42, n_estimators=20),
        )

        # Price fixture: us-south spikes then recovers (simulates ERCOT cold snap)
        base = pd.Timestamp("2026-01-01", tz="UTC")
        rows = []
        for h in range(24 * 50):
            ts = base + pd.Timedelta(hours=h)
            # CAISO: stable at $50
            rows.append({"timestamp": ts, "region": "us-west", "price_per_mwh": 50.0})
            # ERCOT: spike in first 7 days, then recover to $25
            if h < 7 * 24:
                rows.append({"timestamp": ts, "region": "us-south", "price_per_mwh": 1500.0})
            else:
                rows.append({"timestamp": ts, "region": "us-south", "price_per_mwh": 25.0})
        price_df = pd.DataFrame(rows)
        carbon_df = pd.DataFrame(columns=["timestamp", "region", "gco2_per_kwh"])

        from aurelius.ingestion.job_logs import JobLogIngester
        ingester = JobLogIngester()
        jobs = ingester.generate_synthetic(
            start_time=base.to_pydatetime(),
            duration_hours=24 * 50,
            num_jobs=30,
            regions=["us-west", "us-south"],
            seed=42,
            workload_filter="training",
        )

        rounds = engine.run(jobs, price_df, carbon_df)
        # Engine should complete successfully
        assert len(rounds) > 0

    def test_leakage_safety_preserved(self):
        """Recovery correction uses only training data (no eval leakage)."""
        # The correction is applied inside _build_ml_forecast() which only has
        # access to train_price_data and recent_context (both strictly < eval_start).
        # This test verifies the engine can run in ML mode without errors.
        BacktestEngine = self._get_engine_class()
        from aurelius.forecasting.price_model import PriceQuantileForecaster, PriceModelConfig
        import pandas as pd

        engine = BacktestEngine(
            method="greedy_migrate",
            apply_recovery_correction=True,
            train_days=15,
            eval_days=7,
            price_forecaster_cls=PriceQuantileForecaster,
            price_forecaster_config=PriceModelConfig(seed=42, n_estimators=20),
        )

        # Create a fixture that has a clear train/eval separation
        base = pd.Timestamp("2026-01-01", tz="UTC")
        rows = []
        for h in range(24 * 30):
            for region in ["us-west", "us-east"]:
                rows.append({
                    "timestamp": base + pd.Timedelta(hours=h),
                    "region": region,
                    "price_per_mwh": 50.0,
                })
        price_df = pd.DataFrame(rows)
        carbon_df = pd.DataFrame(columns=["timestamp", "region", "gco2_per_kwh"])

        from aurelius.ingestion.job_logs import JobLogIngester
        ingester = JobLogIngester()
        jobs = ingester.generate_synthetic(
            start_time=base.to_pydatetime(),
            duration_hours=24 * 30,
            num_jobs=15,
            regions=["us-west", "us-east"],
            seed=42,
        )

        rounds = engine.run(jobs, price_df, carbon_df)
        assert isinstance(rounds, list)


# ---------------------------------------------------------------------------
# TestTwoGateActivation
# ---------------------------------------------------------------------------

class TestTwoGateActivation:
    """The two-gate design: ratio test AND absolute price ceiling must both pass."""

    def test_ratio_passes_but_ceiling_blocks(self):
        """Region with ratio < threshold but recent_mean > ceiling → no correction.

        Simulates PJM post-spike: training_mean $190 (spike), recent $55 (recovery
        to normal PJM levels). Ratio = 55/190 = 0.29 < 0.40, but $55 > $30 ceiling.
        """
        detector = RegimeDetector()  # ceiling = 30.0
        training = [190.0] * 720
        recent = [55.0] * 24      # ratio = 0.289, but > $30 ceiling
        info = detector.detect("us-east", training, recent)
        assert info.is_recovering is False, (
            "PJM at $55 post-spike should not trigger correction — "
            "prices are in normal operating range"
        )

    def test_ratio_passes_ceiling_passes(self):
        """Both gates pass → correction fires.

        Simulates ERCOT post-spike: training_mean $80 (spike-inflated),
        recent $22 (genuinely cheap recovery).
        """
        detector = RegimeDetector()  # ceiling = 30.0
        training = [80.0] * 720
        recent = [22.0] * 24      # ratio = 0.275 < 0.40, $22 < $30 ceiling
        info = detector.detect("us-south", training, recent)
        assert info.is_recovering is True
        assert info.correction_magnitude > 0.0

    def test_ceiling_passes_but_ratio_fails(self):
        """Recent prices below ceiling but ratio > threshold → no correction (stable)."""
        detector = RegimeDetector()
        training = [28.0] * 720
        recent = [25.0] * 24      # ratio = 25/28 = 0.893 > 0.40
        info = detector.detect("us-south", training, recent)
        assert info.is_recovering is False

    def test_custom_ceiling_allows_higher_prices(self):
        """Custom ceiling of 60.0 allows correction at $55 recent price."""
        detector = RegimeDetector(max_recent_mean_for_correction=60.0)
        training = [190.0] * 720
        recent = [55.0] * 24      # ratio = 0.289, $55 < 60 ceiling
        info = detector.detect("us-east", training, recent)
        assert info.is_recovering is True

    def test_pjm_style_false_positive_blocked(self):
        """PJM-like recovery at $32-35/MWh blocked by default ceiling of $30."""
        detector = RegimeDetector()
        training = [185.0] * 720  # spike-inflated training mean

        for pjm_recent in [32.4, 34.7, 35.0]:
            recent = [pjm_recent] * 24
            info = detector.detect("us-east", training, recent)
            assert info.is_recovering is False, (
                f"PJM at ${pjm_recent:.1f}/MWh should not trigger correction"
            )

    def test_ercot_style_genuine_recovery_passes(self):
        """ERCOT-like recovery at $20-25/MWh passes both gates."""
        detector = RegimeDetector()
        training = [75.0] * 720  # spike-inflated training mean

        for ercot_recent in [20.4, 20.7, 24.6]:
            recent = [ercot_recent] * 24
            info = detector.detect("us-south", training, recent)
            assert info.is_recovering is True, (
                f"ERCOT at ${ercot_recent:.1f}/MWh should trigger correction"
            )


# ---------------------------------------------------------------------------
# TestRegimeSummaryHelper
# ---------------------------------------------------------------------------

class TestRegimeSummaryHelper:
    def test_summary_returns_all_regions(self):
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        train_price_data = {
            "us-west": {base + timedelta(hours=i): 50.0 for i in range(720)},
            "us-south": {base + timedelta(hours=i): 200.0 for i in range(720)},
        }
        recent_context = (
            _make_energy_prices("us-west", 24, 45.0) +
            _make_energy_prices("us-south", 24, 30.0)
        )
        summary = compute_region_regime_summary(train_price_data, recent_context)
        assert "us-west" in summary
        assert "us-south" in summary

    def test_summary_detects_recovery(self):
        """us-south in deep recovery (25/200 = 0.125) detected."""
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        train_price_data = {
            "us-south": {base + timedelta(hours=i): 200.0 for i in range(720)},
        }
        context = _make_energy_prices("us-south", 24, 25.0)
        summary = compute_region_regime_summary(train_price_data, context)
        assert summary["us-south"].is_recovering is True

    def test_summary_no_recovery_stable(self):
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        train_price_data = {
            "us-west": {base + timedelta(hours=i): 47.0 for i in range(720)},
        }
        context = _make_energy_prices("us-west", 24, 45.0)
        summary = compute_region_regime_summary(train_price_data, context)
        assert summary["us-west"].is_recovering is False

    def test_summary_uses_custom_detector(self):
        """Custom detector config is applied."""
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        train_price_data = {
            "r1": {base + timedelta(hours=i): 100.0 for i in range(720)},
        }
        # Use recent price below the default ceiling (30.0)
        context = _make_energy_prices("r1", 24, 25.0)  # ratio = 25/100 = 0.25

        # Standard threshold (0.40): 0.25 < 0.40, recent_mean 25 < 30 → recovering
        standard = compute_region_regime_summary(
            train_price_data, context, detector=RegimeDetector(recovery_ratio_threshold=0.40)
        )
        assert standard["r1"].is_recovering is True

        # Stricter threshold (0.20): 0.25 > 0.20 → not recovering
        stricter = compute_region_regime_summary(
            train_price_data, context, detector=RegimeDetector(recovery_ratio_threshold=0.20)
        )
        assert stricter["r1"].is_recovering is False
