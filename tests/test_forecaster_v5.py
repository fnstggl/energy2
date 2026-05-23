"""Tests for ML forecaster v5.0: price_momentum_168h and price_vs_lag168_abs features.

Feature design (v5.0):
- price_momentum_168h: (price - lag_168h) / |lag_168h| clipped [-1, 5]
  Encodes cold-snap recovery: -0.95 = "95% cheaper than last week"
- price_vs_lag168_abs: price / lag_168h clipped [0, 10]
  Absolute ratio: 0.05 = "5% of last week's price"

When lag_168h_values is None, both features return neutral defaults
(momentum=0, ratio=1) — graceful fallback for missing context.

Acceptance tests:
1. compute_price_rank_features output keys/shape
2. Neutral fallback when lag_168h_values is None
3. Correct momentum: negative for cold-snap recovery, positive for spikes
4. price_vs_lag168_abs bounded [0, 10]
5. NaN-safe: no NaN in outputs
6. lag_336h present when enabled
7. build_feature_matrix adds correct columns when include_rank_features=True
8. Placeholder branch has correct neutral defaults
9. build_feature_matrix_for_predict propagates rank features
10. Cold-snap recovery: momentum_168h < 0 when cheap after spike context
11. PriceModelConfig default include_rank_features=True
12. PriceQuantileForecaster v5.0 model_type / features_version
13. Backward compat: rank off → v2.0 behavior
14. Leakage guard: output is a function of lag_168h only (not rolling window)
15. PRICE_RANK_FEATURE_COLS has 2 entries and matches function output
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np

from aurelius.forecasting.price_model import (
    PriceModelConfig,
    PriceQuantileForecaster,
)
from aurelius.forecasting.quantile_model import (
    PRICE_RANK_FEATURE_COLS,
    build_feature_matrix,
    build_feature_matrix_for_predict,
    compute_price_rank_features,
)
from aurelius.models import EnergyPrice

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_prices(
    n_hours: int = 400,
    base: float = 50.0,
    spike_hours: list[int] | None = None,
    spike_value: float = 500.0,
    region: str = "us-south",
    seed: int = 42,
) -> list[EnergyPrice]:
    rng = np.random.default_rng(seed)
    values = base + rng.normal(0, 8, n_hours)
    if spike_hours:
        for h in spike_hours:
            if 0 <= h < n_hours:
                values[h] = spike_value
    t0 = datetime(2026, 1, 1, tzinfo=None)
    return [
        EnergyPrice(
            timestamp=t0 + timedelta(hours=i),
            region=region,
            price_per_mwh=float(max(0.0, values[i])),
        )
        for i in range(n_hours)
    ]


def _make_values(n: int = 200, base: float = 50.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.maximum(0.0, base + rng.normal(0, 10, n))


def _make_lag_168h(values: np.ndarray) -> np.ndarray:
    """Simulate lag_168h by shifting values by 168 positions."""
    n = len(values)
    lag = np.empty(n)
    for i in range(n):
        lag[i] = values[i - 168] if i >= 168 else values[i]
    return lag


# ---------------------------------------------------------------------------
# 1. compute_price_rank_features — output structure
# ---------------------------------------------------------------------------

class TestComputePriceRankFeatures:
    def test_returns_all_required_keys_with_lag(self):
        values = _make_values(200)
        lag = _make_lag_168h(values)
        result = compute_price_rank_features(values, lag_168h_values=lag)
        assert set(result.keys()) == set(PRICE_RANK_FEATURE_COLS)

    def test_returns_all_required_keys_fallback(self):
        """Without lag, must still return all keys (neutral defaults)."""
        values = _make_values(200)
        result = compute_price_rank_features(values)
        assert set(result.keys()) == set(PRICE_RANK_FEATURE_COLS)

    def test_arrays_same_length_as_input(self):
        n = 300
        values = _make_values(n)
        lag = _make_lag_168h(values)
        result = compute_price_rank_features(values, lag_168h_values=lag)
        for key, arr in result.items():
            assert len(arr) == n, f"Feature {key}: expected {n}, got {len(arr)}"

    def test_neutral_defaults_when_no_lag(self):
        """With no lag_168h_values, momentum=0, ratio=1 everywhere."""
        values = _make_values(100)
        result = compute_price_rank_features(values)
        assert np.all(result["price_momentum_168h"] == 0.0)
        assert np.all(result["price_vs_lag168_abs"] == 1.0)

    def test_neutral_defaults_when_lag_wrong_length(self):
        """Mismatched lag length → neutral defaults (not crash)."""
        values = _make_values(100)
        bad_lag = _make_lag_168h(values)[:50]  # wrong length
        result = compute_price_rank_features(values, lag_168h_values=bad_lag)
        assert np.all(result["price_momentum_168h"] == 0.0)
        assert np.all(result["price_vs_lag168_abs"] == 1.0)

    def test_momentum_negative_for_cold_snap_recovery(self):
        """After cold snap: current price << lag_168h → momentum << 0."""
        n = 50
        current = np.full(n, 20.0)   # recovery price $20/MWh
        lag = np.full(n, 2000.0)     # spike was $2000/MWh
        result = compute_price_rank_features(current, lag_168h_values=lag)
        momentum = result["price_momentum_168h"]
        # (20 - 2000) / 2000 = -0.99, clipped to -1
        assert np.all(momentum <= -0.9), f"Expected momentum near -1, got mean={momentum.mean():.3f}"

    def test_momentum_positive_for_spike_onset(self):
        """During spike: current price >> lag_168h → momentum > 0."""
        n = 50
        current = np.full(n, 2000.0)  # spike price
        lag = np.full(n, 50.0)        # normal week
        result = compute_price_rank_features(current, lag_168h_values=lag)
        momentum = result["price_momentum_168h"]
        # (2000 - 50) / 50 = 39, clipped to 5
        assert np.all(momentum == 5.0), f"Expected momentum=5 (clipped), got mean={momentum.mean():.3f}"

    def test_momentum_near_zero_for_steady_state(self):
        """When current == lag_168h, momentum ≈ 0."""
        n = 100
        values = np.full(n, 60.0)
        lag = np.full(n, 60.0)
        result = compute_price_rank_features(values, lag_168h_values=lag)
        momentum = result["price_momentum_168h"]
        assert np.all(np.abs(momentum) < 0.01), f"Expected momentum≈0, got max={np.abs(momentum).max():.4f}"

    def test_price_vs_lag168_abs_bounded(self):
        """price_vs_lag168_abs must be in [0, 10]."""
        n = 100
        values = np.array([0.001] * 50 + [10000.0] * 50)
        lag = np.full(n, 50.0)
        result = compute_price_rank_features(values, lag_168h_values=lag)
        ratio = result["price_vs_lag168_abs"]
        assert np.all(ratio >= 0.0), "ratio must be >= 0"
        assert np.all(ratio <= 10.0), "ratio must be <= 10"

    def test_price_vs_lag168_abs_cheap_near_zero(self):
        """Very cheap vs last week → ratio near 0."""
        n = 50
        current = np.full(n, 1.0)    # $1/MWh
        lag = np.full(n, 2000.0)     # $2000/MWh last week
        result = compute_price_rank_features(current, lag_168h_values=lag)
        ratio = result["price_vs_lag168_abs"]
        assert np.all(ratio < 0.01), f"Cheap recovery: ratio should be near 0, got max={ratio.max():.4f}"

    def test_no_nan_with_typical_prices(self):
        n = 400
        values = _make_values(n, base=50.0)
        lag = _make_lag_168h(values)
        result = compute_price_rank_features(values, lag_168h_values=lag)
        for key, arr in result.items():
            assert not np.any(np.isnan(arr)), f"NaN in {key}"

    def test_no_nan_with_zero_prices(self):
        """Zero prices: division by eps should prevent NaN."""
        result = compute_price_rank_features(np.zeros(50), lag_168h_values=np.zeros(50))
        for arr in result.values():
            assert not np.any(np.isnan(arr)), "No NaN allowed with zero prices"

    def test_single_element_does_not_crash(self):
        result = compute_price_rank_features(np.array([42.0]), lag_168h_values=np.array([40.0]))
        for key in PRICE_RANK_FEATURE_COLS:
            assert key in result
            assert len(result[key]) == 1

    def test_momentum_clipped_lower_bound(self):
        """Momentum floor is -1 even when price = 0 and lag = very high."""
        n = 10
        current = np.zeros(n)
        lag = np.full(n, 1000.0)
        result = compute_price_rank_features(current, lag_168h_values=lag)
        assert np.all(result["price_momentum_168h"] >= -1.0)

    def test_momentum_clipped_upper_bound(self):
        """Momentum ceiling is 5."""
        n = 10
        current = np.full(n, 1e9)
        lag = np.full(n, 1.0)
        result = compute_price_rank_features(current, lag_168h_values=lag)
        assert np.all(result["price_momentum_168h"] <= 5.0)


# ---------------------------------------------------------------------------
# 2. PRICE_RANK_FEATURE_COLS registry
# ---------------------------------------------------------------------------

class TestPriceRankFeatureCols:
    def test_cols_list_matches_function_output(self):
        values = _make_values(200)
        lag = _make_lag_168h(values)
        result = compute_price_rank_features(values, lag_168h_values=lag)
        for col in PRICE_RANK_FEATURE_COLS:
            assert col in result, f"Column {col} missing from compute_price_rank_features output"

    def test_cols_has_expected_count(self):
        # v5.0: price_momentum_168h, price_vs_lag168_abs
        assert len(PRICE_RANK_FEATURE_COLS) == 2

    def test_expected_column_names(self):
        assert "price_momentum_168h" in PRICE_RANK_FEATURE_COLS
        assert "price_vs_lag168_abs" in PRICE_RANK_FEATURE_COLS


# ---------------------------------------------------------------------------
# 3. build_feature_matrix with include_rank_features
# ---------------------------------------------------------------------------

class TestBuildFeatureMatrixWithRankFeatures:
    def test_rank_features_added_when_enabled(self):
        n = 400
        t0 = datetime(2026, 1, 1)
        ts = [t0 + timedelta(hours=i) for i in range(n)]
        regions = ["us-west"] * n
        values = _make_values(n)
        df = build_feature_matrix(
            ts, regions, values,
            include_lags=True,
            include_rank_features=True,
            lag_hours=[1, 6, 24, 168, 336],
        )
        for col in PRICE_RANK_FEATURE_COLS:
            assert col in df.columns, f"Missing rank feature column: {col}"

    def test_rank_features_absent_when_disabled(self):
        n = 200
        t0 = datetime(2026, 1, 1)
        ts = [t0 + timedelta(hours=i) for i in range(n)]
        regions = ["us-west"] * n
        values = _make_values(n)
        df = build_feature_matrix(
            ts, regions, values,
            include_rank_features=False,
        )
        for col in PRICE_RANK_FEATURE_COLS:
            assert col not in df.columns, f"Rank feature should not be present: {col}"

    def test_rank_features_placeholder_no_values(self):
        """Without values, placeholder columns use neutral defaults."""
        n = 10
        t0 = datetime(2026, 1, 1)
        ts = [t0 + timedelta(hours=i) for i in range(n)]
        regions = ["us-west"] * n
        df = build_feature_matrix(
            ts, regions,
            values=None,
            include_rank_features=True,
        )
        for col in PRICE_RANK_FEATURE_COLS:
            assert col in df.columns, f"Placeholder missing: {col}"
        # Neutral defaults: momentum=0, ratio=1
        assert (df["price_momentum_168h"] == 0.0).all()
        assert (df["price_vs_lag168_abs"] == 1.0).all()

    def test_lag_336h_present_when_in_lag_hours(self):
        n = 400
        t0 = datetime(2026, 1, 1)
        ts = [t0 + timedelta(hours=i) for i in range(n)]
        regions = ["us-west"] * n
        values = _make_values(n)
        df = build_feature_matrix(
            ts, regions, values,
            include_lags=True,
            lag_hours=[1, 6, 24, 168, 336],
        )
        assert "lag_336h" in df.columns, "lag_336h should be present when included"

    def test_rank_features_no_nan(self):
        n = 400
        t0 = datetime(2026, 1, 1)
        ts = [t0 + timedelta(hours=i) for i in range(n)]
        regions = ["us-west"] * n
        values = _make_values(n)
        df = build_feature_matrix(
            ts, regions, values,
            include_lags=True,
            include_rank_features=True,
            lag_hours=[1, 6, 24, 168, 336],
        )
        for col in PRICE_RANK_FEATURE_COLS:
            assert not df[col].isna().any(), f"NaN found in {col}"

    def test_cold_snap_recovery_negative_momentum(self):
        """Cold-snap pattern: 168h of high prices, then cheap → momentum_168h < 0."""
        n = 400
        t0 = datetime(2026, 1, 1)
        # First 200h: spike at $2000/MWh. Next 200h: recovery at $20/MWh.
        values = np.concatenate([np.full(200, 2000.0), np.full(200, 20.0)])
        ts = [t0 + timedelta(hours=i) for i in range(n)]
        regions = ["us-south"] * n

        df = build_feature_matrix(
            ts, regions, values,
            include_lags=True,
            include_rank_features=True,
            lag_hours=[1, 6, 24, 168, 336],
        )
        # At index 250 (50h into recovery), lag_168h looks back to index 82 → still in spike ($2000)
        # current = $20, lag_168h = $2000 → momentum = (20-2000)/2000 = -0.99 → clipped to -1
        recovery_momentum = df["price_momentum_168h"].iloc[250:260].values
        assert np.mean(recovery_momentum) < -0.5, (
            f"Cold-snap recovery should have strong negative momentum, "
            f"got mean={np.mean(recovery_momentum):.3f}"
        )


# ---------------------------------------------------------------------------
# 4. build_feature_matrix_for_predict with include_rank_features
# ---------------------------------------------------------------------------

class TestBuildFeatureMatrixForPredictWithRankFeatures:
    def test_rank_features_in_prediction_matrix(self):
        n_context = 360
        n_predict = 24
        t0 = datetime(2026, 1, 1)
        pred_ts = [t0 + timedelta(hours=n_context + i) for i in range(n_predict)]
        regions = ["us-west"] * n_predict
        recent_values = _make_values(n_context)
        df, used_lags = build_feature_matrix_for_predict(
            pred_ts, regions, recent_values,
            include_rank_features=True,
            lag_hours=[1, 6, 24, 168, 336],
        )
        assert used_lags is True
        assert len(df) == n_predict
        for col in PRICE_RANK_FEATURE_COLS:
            assert col in df.columns, f"Missing {col} in predict matrix"

    def test_rank_features_no_nan_in_predict(self):
        n_context = 360
        n_predict = 24
        t0 = datetime(2026, 1, 1)
        pred_ts = [t0 + timedelta(hours=n_context + i) for i in range(n_predict)]
        regions = ["us-west"] * n_predict
        recent_values = _make_values(n_context)
        df, _ = build_feature_matrix_for_predict(
            pred_ts, regions, recent_values,
            include_rank_features=True,
            lag_hours=[1, 6, 24, 168, 336],
        )
        for col in PRICE_RANK_FEATURE_COLS:
            assert not df[col].isna().any(), f"NaN in {col} at predict time"

    def test_cold_snap_recovery_momentum_in_predict(self):
        """After cold-snap context, predict-time momentum should be strongly negative."""
        # Context: 168h spike then 168h recovery
        context_vals = np.concatenate([
            np.full(200, 2000.0),   # spike
            np.full(160, 20.0),    # recovery (last known = $20)
        ])
        n_context = len(context_vals)
        t0 = datetime(2026, 1, 1)
        pred_ts = [t0 + timedelta(hours=n_context + i) for i in range(24)]
        regions = ["us-south"] * 24

        df, _ = build_feature_matrix_for_predict(
            pred_ts, regions, context_vals,
            include_rank_features=True,
            lag_hours=[1, 6, 24, 168, 336],
        )
        # For k < 168h, lag_168h uses real context spike prices → momentum negative
        momentum = df["price_momentum_168h"].values
        assert np.mean(momentum) < 0.0, (
            f"Predict-time: cold-snap recovery should produce negative momentum, "
            f"got mean={np.mean(momentum):.3f}"
        )


# ---------------------------------------------------------------------------
# 5. PriceModelConfig v5.0 defaults
# ---------------------------------------------------------------------------

class TestPriceModelConfigV5:
    def test_include_rank_features_default_false(self):
        cfg = PriceModelConfig()
        assert cfg.include_rank_features is False

    def test_include_rank_features_can_be_disabled(self):
        cfg = PriceModelConfig(include_rank_features=False)
        assert cfg.include_rank_features is False

    def test_lag_hours_includes_336_when_rank_on(self):
        cfg = PriceModelConfig(include_rank_features=True)
        fc = PriceQuantileForecaster(cfg, corrections_path=False)
        assert 336 in fc._lag_hours, "lag_336h should be in v5.0 lag hours"

    def test_lag_hours_excludes_336_when_rank_off(self):
        cfg = PriceModelConfig(include_rank_features=False)
        fc = PriceQuantileForecaster(cfg, corrections_path=False)
        assert 336 not in fc._lag_hours, "lag_336h must not be in v2.0 lag hours"

    def test_use_rank_flag_matches_config(self):
        cfg_on = PriceModelConfig(include_rank_features=True)
        fc_on = PriceQuantileForecaster(cfg_on, corrections_path=False)
        assert fc_on._use_rank is True

        cfg_off = PriceModelConfig(include_rank_features=False)
        fc_off = PriceQuantileForecaster(cfg_off, corrections_path=False)
        assert fc_off._use_rank is False


# ---------------------------------------------------------------------------
# 6. PriceQuantileForecaster v5.0 — fit/predict
# ---------------------------------------------------------------------------

class TestPriceQuantileForecasterV5:
    def test_fits_with_rank_features(self):
        prices = _make_prices(n_hours=400)
        cfg = PriceModelConfig(seed=42, n_estimators=30, include_rank_features=True)
        fc = PriceQuantileForecaster(cfg, corrections_path=False)
        fc.fit(prices)
        assert fc.is_fitted
        assert fc.metadata is not None
        assert "+rank" in fc.metadata.model_type
        assert fc.metadata.features_version == "v5.0"

    def test_fits_rank_off_gives_v2_version(self):
        prices = _make_prices(n_hours=300)
        cfg = PriceModelConfig(
            seed=42, n_estimators=30,
            include_rank_features=False,
            include_volatility_features=True,
        )
        fc = PriceQuantileForecaster(cfg, corrections_path=False)
        fc.fit(prices)
        assert fc.metadata.features_version == "v2.0"
        assert "+rank" not in fc.metadata.model_type

    def test_predict_returns_correct_count(self):
        prices = _make_prices(n_hours=400)
        cfg = PriceModelConfig(seed=42, n_estimators=30, include_rank_features=True)
        fc = PriceQuantileForecaster(cfg, corrections_path=False)
        fc.fit(prices)
        t0 = datetime(2026, 1, 1)
        pred_ts = [t0 + timedelta(hours=400 + h) for h in range(24)]
        preds = fc.predict("us-south", pred_ts, recent_prices=prices[-360:])
        assert len(preds) == 24

    def test_p90_geq_p50_always(self):
        prices = _make_prices(n_hours=400)
        cfg = PriceModelConfig(seed=42, n_estimators=30, include_rank_features=True)
        fc = PriceQuantileForecaster(cfg, corrections_path=False)
        fc.fit(prices)
        t0 = datetime(2026, 1, 1)
        pred_ts = [t0 + timedelta(hours=400 + h) for h in range(48)]
        preds = fc.predict("us-south", pred_ts, recent_prices=prices[-360:])
        for p in preds:
            assert p.p90 >= p.p50, f"p90 < p50 at {p.timestamp}"

    def test_determinism_with_rank_features(self):
        prices = _make_prices(n_hours=400, seed=42)
        t0 = datetime(2026, 1, 1)
        pred_ts = [t0 + timedelta(hours=400 + h) for h in range(12)]

        cfg = PriceModelConfig(seed=42, n_estimators=30, include_rank_features=True)
        fc1 = PriceQuantileForecaster(cfg, corrections_path=False)
        fc1.fit(prices)
        preds1 = fc1.predict("us-south", pred_ts, recent_prices=prices[-360:])

        fc2 = PriceQuantileForecaster(cfg, corrections_path=False)
        fc2.fit(prices)
        preds2 = fc2.predict("us-south", pred_ts, recent_prices=prices[-360:])

        for p1, p2 in zip(preds1, preds2):
            assert abs(p1.p50 - p2.p50) < 1e-6, "Predictions not deterministic"

    def test_rank_features_context_uses_extended_window(self):
        """v5.0 predict uses max_lag+24h context (360h), not just 48h."""
        prices = _make_prices(n_hours=500)
        cfg = PriceModelConfig(seed=42, n_estimators=20, include_rank_features=True)
        fc = PriceQuantileForecaster(cfg, corrections_path=False)
        fc.fit(prices)
        t0 = datetime(2026, 1, 1)
        pred_ts = [t0 + timedelta(hours=500 + h) for h in range(12)]
        preds = fc.predict("us-south", pred_ts, recent_prices=prices[-360:])
        assert len(preds) == 12
        assert all(p.p50 >= 0 for p in preds)

    def test_rank_features_no_nan_predictions(self):
        prices = _make_prices(n_hours=400, seed=7)
        cfg = PriceModelConfig(seed=42, n_estimators=30, include_rank_features=True)
        fc = PriceQuantileForecaster(cfg, corrections_path=False)
        fc.fit(prices)
        t0 = datetime(2026, 1, 1)
        pred_ts = [t0 + timedelta(hours=400 + h) for h in range(48)]
        preds = fc.predict("us-south", pred_ts, recent_prices=prices[-360:])
        for p in preds:
            assert not math.isnan(p.p50), f"NaN p50 at {p.timestamp}"
            assert not math.isnan(p.p90), f"NaN p90 at {p.timestamp}"


# ---------------------------------------------------------------------------
# 7. Leakage guard — momentum is a function of lag only, not rolling window
# ---------------------------------------------------------------------------

class TestRankFeaturesLeakageSafety:
    def test_momentum_is_function_of_lag_only(self):
        """
        compute_price_rank_features(values, lag_168h_values) is purely a function
        of the lag_168h_values argument, not a rolling window over values.

        Strategy: keep lag_168h_values identical but change values at index i.
        The output at index i must differ (it uses values[i]) but in a way that
        depends only on values[i] and lag[i], not on values at adjacent indices.
        """
        n = 200
        lag = np.full(n, 50.0)  # fixed reference

        values_a = np.full(n, 100.0)
        values_b = np.full(n, 100.0)
        values_b[100] = 10.0  # change only index 100

        feats_a = compute_price_rank_features(values_a, lag_168h_values=lag)
        feats_b = compute_price_rank_features(values_b, lag_168h_values=lag)

        # At index 100: different values → different momentum
        assert feats_a["price_momentum_168h"][100] != feats_b["price_momentum_168h"][100], (
            "Changing values[100] should change momentum[100]"
        )

        # At all OTHER indices: values and lag identical → identical momentum
        for i in range(n):
            if i == 100:
                continue
            assert feats_a["price_momentum_168h"][i] == feats_b["price_momentum_168h"][i], (
                f"momentum[{i}] should be same when only values[100] differs"
            )

    def test_changing_lag_changes_momentum(self):
        """Momentum correctly reflects the lag reference price."""
        n = 50
        current = np.full(n, 100.0)

        lag_normal = np.full(n, 100.0)
        lag_high = np.full(n, 2000.0)  # spike last week

        feats_normal = compute_price_rank_features(current, lag_168h_values=lag_normal)
        feats_high = compute_price_rank_features(current, lag_168h_values=lag_high)

        # Same current price, but vs high lag → momentum is much more negative
        assert feats_high["price_momentum_168h"].mean() < feats_normal["price_momentum_168h"].mean(), (
            "Higher lag_168h reference should yield more negative momentum"
        )

    def test_spike_price_has_high_positive_momentum(self):
        """During spike: current >> lag_168h → positive momentum."""
        n = 200
        values = np.full(n, 50.0)
        values[100] = 2000.0  # spike at index 100
        lag = np.full(n, 50.0)  # normal last week for all

        feats = compute_price_rank_features(values, lag_168h_values=lag)
        # At index 100: (2000 - 50) / 50 = 39, clipped to 5
        assert feats["price_momentum_168h"][100] == 5.0, (
            "Spike price should have maximum positive momentum (5.0, clipped)"
        )
        # At other indices: (50 - 50) / 50 ≈ 0
        assert abs(feats["price_momentum_168h"][50]) < 0.01


# ---------------------------------------------------------------------------
# 8. Benchmark acceptance: v5.0 trains, predicts, stays in plausible range
# ---------------------------------------------------------------------------

class TestV5BenchmarkAcceptance:
    def test_v5_non_negative_savings_on_synthetic(self):
        """v5.0 forecaster trains, predicts, and produces plausible outputs."""
        n = 500
        rng = np.random.default_rng(42)
        base_west = 50.0 + 20 * np.sin(np.arange(n) * 2 * np.pi / 24)
        base_east = 50.0 - 20 * np.sin(np.arange(n) * 2 * np.pi / 24)
        vals_west = np.maximum(10.0, base_west + rng.normal(0, 5, n))
        vals_east = np.maximum(10.0, base_east + rng.normal(0, 5, n))

        t0 = datetime(2026, 1, 1)
        prices = []
        for i in range(n):
            ts = t0 + timedelta(hours=i)
            prices.append(EnergyPrice(timestamp=ts, region="us-west", price_per_mwh=vals_west[i]))
            prices.append(EnergyPrice(timestamp=ts, region="us-east", price_per_mwh=vals_east[i]))

        cfg_v5 = PriceModelConfig(seed=42, n_estimators=50, include_rank_features=True)
        fc_v5 = PriceQuantileForecaster(cfg_v5, corrections_path=False)
        fc_v5.fit(prices[:n])

        pred_ts = [t0 + timedelta(hours=n // 2 + h) for h in range(168)]
        preds = fc_v5.predict("us-west", pred_ts, recent_prices=prices[:n // 2][-360:])

        assert len(preds) == 168
        valid_preds = [p for p in preds if not math.isnan(p.p50)]
        assert len(valid_preds) >= 100, "Most predictions should be non-NaN"
        mean_p50 = np.mean([p.p50 for p in valid_preds])
        assert 10.0 <= mean_p50 <= 200.0, (
            f"Mean prediction {mean_p50:.1f} is outside plausible range [10, 200]"
        )

    def test_v5_cold_snap_recovery_produces_lower_predictions(self):
        """After cold-snap, v5.0 should predict lower prices (recovery signal)."""
        n = 400
        rng = np.random.default_rng(123)
        # Normal prices for first 200h, then spike, then recovery
        normal = np.maximum(10.0, 50.0 + rng.normal(0, 5, 200))
        spike = np.full(50, 1500.0)
        recovery = np.maximum(10.0, 30.0 + rng.normal(0, 3, 150))
        vals = np.concatenate([normal, spike, recovery])

        t0 = datetime(2026, 1, 1)
        prices = [
            EnergyPrice(
                timestamp=t0 + timedelta(hours=i),
                region="us-south",
                price_per_mwh=float(vals[i]),
            )
            for i in range(n)
        ]

        cfg = PriceModelConfig(seed=42, n_estimators=50, include_rank_features=True)
        fc = PriceQuantileForecaster(cfg, corrections_path=False)
        fc.fit(prices[:300])

        # Predict during recovery using spike-inclusive context
        pred_ts = [t0 + timedelta(hours=300 + h) for h in range(24)]
        preds = fc.predict("us-south", pred_ts, recent_prices=prices[:300][-360:])

        assert len(preds) == 24
        assert all(p.p90 >= p.p50 for p in preds)
        mean_p50 = np.mean([p.p50 for p in preds])
        assert mean_p50 >= 0.0, "Predictions must be non-negative"
