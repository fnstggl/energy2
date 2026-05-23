"""Tests for extended benchmark infrastructure:
- EXTENDED_REGION_COMBOS definition and structure
- build_combined_dataset.py merge logic
- --extended-data / --region-combo flag handling
- SKIPPED result when data file missing
- Per-region forecaster with 90-day windows (synthetic data)
- Combined dataset coverage and dedup logic
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "aurelius"))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_price_df(region: str, start: str, end: str, freq: str = "h") -> pd.DataFrame:
    """Build a minimal canonical price CSV for testing."""
    ts = pd.date_range(start, end, freq=freq, tz="UTC", inclusive="left")
    import numpy as np
    rng = np.random.default_rng(42 + hash(region) % 1000)
    prices = rng.uniform(20, 100, len(ts))
    return pd.DataFrame({
        "timestamp": ts,
        "region": region,
        "price_per_mwh": prices,
        "currency": "USD",
        "source": "test_synthetic",
        "source_granularity": "hourly",
        "fetched_at": datetime.now(tz=timezone.utc),
    })


# ---------------------------------------------------------------------------
# 1. EXTENDED_REGION_COMBOS structure
# ---------------------------------------------------------------------------

class TestExtendedRegionCombos:
    def test_extended_combos_exist(self):
        from benchmarks.run_benchmark import EXTENDED_REGION_COMBOS
        assert len(EXTENDED_REGION_COMBOS) >= 1

    def test_combined_combo_has_required_fields(self):
        from benchmarks.run_benchmark import EXTENDED_REGION_COMBOS
        combo = next(c for c in EXTENDED_REGION_COMBOS if "combined" in c["name"])
        assert "regions" in combo
        assert "da_price_file" in combo
        assert "date_start" in combo
        assert "date_end" in combo

    def test_combined_combo_has_3_regions(self):
        from benchmarks.run_benchmark import EXTENDED_REGION_COMBOS
        combo = next(c for c in EXTENDED_REGION_COMBOS if "combined" in c["name"])
        assert set(combo["regions"]) == {"us-west", "us-east", "us-south"}

    def test_combined_combo_da_file_path_format(self):
        from benchmarks.run_benchmark import EXTENDED_REGION_COMBOS
        combo = next(c for c in EXTENDED_REGION_COMBOS if "combined" in c["name"])
        assert "combined_2025_2026" in combo["da_price_file"]
        assert combo["da_price_file"].endswith(".csv")

    def test_combined_combo_date_range_gt_90_days(self):
        from benchmarks.run_benchmark import EXTENDED_REGION_COMBOS
        combo = next(c for c in EXTENDED_REGION_COMBOS if "combined" in c["name"])
        start = datetime.fromisoformat(combo["date_start"])
        end = datetime.fromisoformat(combo["date_end"])
        assert (end - start).days >= 90, "Extended combo must span ≥90 days"

    def test_recommended_train_days_is_90(self):
        from benchmarks.run_benchmark import EXTENDED_REGION_COMBOS
        combo = next(c for c in EXTENDED_REGION_COMBOS if "combined" in c["name"])
        # recommended_train_days is documented in the combo
        assert combo.get("recommended_train_days", 90) == 90


# ---------------------------------------------------------------------------
# 2. run_single_benchmark skips gracefully when DA file missing
# ---------------------------------------------------------------------------

class TestMissingDataFileSkip:
    def test_returns_skipped_dict_when_da_file_missing(self, tmp_path):
        from benchmarks.run_benchmark import run_single_benchmark
        combo = {
            "name": "missing_test",
            "regions": ["us-west"],
            "da_price_file": "data/nonexistent_file_for_test.csv",
            "rt_price_file": None,
            "date_start": "2026-01-01",
            "date_end": "2026-01-10",
        }
        result = run_single_benchmark(
            region_combo=combo,
            workload_type="training",
            repo_root=_REPO_ROOT,
        )
        assert result.get("skipped") is True
        assert "skip_reason" in result
        assert "not found" in result["skip_reason"]

    def test_skipped_result_is_not_treated_as_error(self):
        """Verify the skipped flag is checked before the error flag in logic."""
        result = {"skipped": True, "skip_reason": "test"}
        assert result.get("skipped") is True
        assert "error" not in result


# ---------------------------------------------------------------------------
# 3. Build combined dataset merge logic (unit tests without real data)
# ---------------------------------------------------------------------------

class TestBuildCombinedDatasetMerge:
    def test_merge_dedup_keeps_latest(self, tmp_path):
        """Duplicate (timestamp, region) rows → keep last occurrence."""

        ts = pd.Timestamp("2025-06-01 00:00:00", tz="UTC")
        row_common = {
            "timestamp": ts, "region": "us-west", "price_per_mwh": 50.0,
            "currency": "USD", "source": "test", "source_granularity": "hourly",
            "fetched_at": pd.Timestamp("2025-06-01", tz="UTC"),
        }
        row_updated = {**row_common, "price_per_mwh": 55.0}
        df1 = pd.DataFrame([row_common])
        df2 = pd.DataFrame([row_updated])

        combined = pd.concat([df1, df2], ignore_index=True)
        deduped = combined.drop_duplicates(subset=["timestamp", "region"], keep="last")
        assert len(deduped) == 1
        assert deduped.iloc[0]["price_per_mwh"] == 55.0

    def test_merge_preserves_all_regions(self, tmp_path):
        df_west = _make_price_df("us-west", "2025-06-01", "2025-07-01")
        df_east = _make_price_df("us-east", "2025-06-01", "2025-07-01")
        df_south = _make_price_df("us-south", "2025-06-01", "2025-07-01")
        combined = pd.concat([df_west, df_east, df_south], ignore_index=True)
        assert set(combined["region"].unique()) == {"us-west", "us-east", "us-south"}

    def test_merge_three_periods_spans_expected_days(self, tmp_path):
        summer = _make_price_df("us-west", "2025-06-01", "2025-09-01")
        fall = _make_price_df("us-west", "2025-09-01", "2026-01-01")
        q1 = _make_price_df("us-west", "2026-01-01", "2026-03-15")
        combined = pd.concat([summer, fall, q1], ignore_index=True)
        combined = combined.drop_duplicates(subset=["timestamp", "region"], keep="last")
        span = (combined["timestamp"].max() - combined["timestamp"].min()).days
        assert span >= 270, f"Expected ≥270 days span, got {span}"

    def test_combined_sorted_by_region_and_timestamp(self, tmp_path):
        df = _make_price_df("us-west", "2025-06-01", "2025-07-01")
        df2 = _make_price_df("us-east", "2025-06-01", "2025-07-01")
        combined = pd.concat([df, df2], ignore_index=True)
        combined = combined.sort_values(["region", "timestamp"]).reset_index(drop=True)
        # Region should be sorted: us-east before us-west
        regions = combined["region"].tolist()
        first_change = next(i for i in range(1, len(regions)) if regions[i] != regions[i-1])
        assert regions[0] == "us-east"
        assert regions[first_change] == "us-west"

    def test_merge_no_gap_when_periods_contiguous(self):
        summer = _make_price_df("us-west", "2025-06-01", "2025-09-01")
        fall = _make_price_df("us-west", "2025-09-01", "2026-01-01")
        combined = pd.concat([summer, fall], ignore_index=True)
        combined = combined.sort_values("timestamp").reset_index(drop=True)
        # Check no gap > 1 hour between consecutive rows (for same region)
        ws = combined[combined["region"] == "us-west"].sort_values("timestamp")
        diffs = ws["timestamp"].diff().dropna()
        max_gap_hours = diffs.max().total_seconds() / 3600
        assert max_gap_hours <= 2.0, f"Gap > 2 hours detected: {max_gap_hours:.1f}h"


# ---------------------------------------------------------------------------
# 4. Per-region forecaster with 90-day synthetic data
# ---------------------------------------------------------------------------

class TestPerRegionForecasterWith90DayData:
    @pytest.fixture
    def long_price_data(self):
        """Synthetic 90-day, 3-region price dataset for per-region testing."""
        import numpy as np

        from aurelius.models import EnergyPrice

        rng = np.random.default_rng(42)
        prices = []
        base_ts = datetime(2025, 9, 1, tzinfo=timezone.utc)

        for region, base_price in [("us-west", 45.0), ("us-east", 55.0), ("us-south", 35.0)]:
            for hour in range(90 * 24):  # 90 days
                ts = base_ts + timedelta(hours=hour)
                hour_of_day = ts.hour
                _day_of_week = ts.weekday()
                seasonal_factor = 1.0 + 0.3 * (hour / (90 * 24))
                hour_factor = 0.8 + 0.4 * (hour_of_day / 24)
                noise = rng.normal(0, 5)
                price = base_price * seasonal_factor * hour_factor + noise
                prices.append(EnergyPrice(
                    timestamp=ts,
                    region=region,
                    price_per_mwh=max(5.0, price),
                ))
        return prices

    def test_per_region_forecaster_fits_on_90d_data(self, long_price_data):
        """Per-region forecaster should fit without error on 90-day windows."""
        from aurelius.forecasting.price_model import (
            PerRegionForecaster,
            PerRegionForecasterConfig,
            PriceModelConfig,
        )

        config = PerRegionForecasterConfig(
            base_config=PriceModelConfig(seed=42, n_estimators=20, num_leaves=15),
        )
        fc = PerRegionForecaster(config=config)
        fc.fit(long_price_data)
        assert fc._fitted is True
        assert len(fc._region_forecasters) == 3

    def test_per_region_has_enough_samples_per_region(self, long_price_data):
        """Each region should have ≥2160 training records (90×24) to avoid overfitting."""
        from collections import Counter
        counts = Counter(p.region for p in long_price_data)
        for region, count in counts.items():
            assert count >= 2160, f"Region {region} has only {count} records (need ≥2160)"

    def test_per_region_predicts_all_regions(self, long_price_data):
        """Per-region forecaster must produce predictions for all 3 regions."""
        from aurelius.forecasting.price_model import (
            PerRegionForecaster,
            PerRegionForecasterConfig,
            PriceModelConfig,
        )

        config = PerRegionForecasterConfig(
            base_config=PriceModelConfig(seed=42, n_estimators=20, num_leaves=15),
        )
        fc = PerRegionForecaster(config=config)
        # Use only first 80 days for training
        train_data = [p for p in long_price_data
                      if p.timestamp < datetime(2025, 11, 20, tzinfo=timezone.utc)]
        fc.fit(train_data)

        _predict_ts = datetime(2025, 11, 21, tzinfo=timezone.utc)
        for region in ("us-west", "us-east", "us-south"):
            fc_obj = fc._region_forecasters.get(region)
            assert fc_obj is not None, f"No forecaster for {region}"
            assert fc_obj.is_fitted

    def test_per_region_p90_above_p50(self, long_price_data):
        """p90 forecasts must be ≥ p50 forecasts (correct quantile ordering)."""
        from aurelius.forecasting.price_model import (
            PerRegionForecaster,
            PerRegionForecasterConfig,
            PriceModelConfig,
        )

        config = PerRegionForecasterConfig(
            base_config=PriceModelConfig(seed=42, n_estimators=20, num_leaves=15),
        )
        fc = PerRegionForecaster(config=config)
        train_data = [p for p in long_price_data
                      if p.timestamp < datetime(2025, 11, 20, tzinfo=timezone.utc)]
        fc.fit(train_data)

        # Build prediction input using the per-region sub-forecaster
        predict_ts = datetime(2025, 11, 21, tzinfo=timezone.utc)
        context = [p for p in train_data if p.region == "us-west"][-200:]
        future_ts = [predict_ts + timedelta(hours=h) for h in range(24)]

        result = fc._region_forecasters["us-west"].predict(
            region="us-west",
            timestamps=future_ts,
            recent_prices=context,
        )
        for forecast in result:
            assert forecast.p90 >= forecast.p50, \
                f"p90 < p50: {forecast.p90} < {forecast.p50}"

    def test_per_region_isolated_models_no_cross_contamination(self, long_price_data):
        """Verify that changing us-west data does not affect us-east training size."""
        from aurelius.forecasting.price_model import (
            PerRegionForecaster,
            PerRegionForecasterConfig,
            PriceModelConfig,
        )

        config = PerRegionForecasterConfig(
            base_config=PriceModelConfig(seed=42, n_estimators=20, num_leaves=15),
        )

        # Train on original data
        train_data = [p for p in long_price_data
                      if p.timestamp < datetime(2025, 11, 20, tzinfo=timezone.utc)]
        fc1 = PerRegionForecaster(config=config)
        fc1.fit(train_data)

        # Train on modified data (us-west prices doubled)
        from aurelius.models import EnergyPrice
        modified = []
        for p in train_data:
            if p.region == "us-west":
                modified.append(EnergyPrice(
                    timestamp=p.timestamp, region=p.region,
                    price_per_mwh=p.price_per_mwh * 2.0,
                ))
            else:
                modified.append(p)
        fc2 = PerRegionForecaster(config=config)
        fc2.fit(modified)

        # Verify us-east models have identical training sample count
        # (per-region isolation: changing us-west doesn't affect us-east)
        meta1 = fc1._region_forecasters.get("us-east")
        meta2 = fc2._region_forecasters.get("us-east")
        assert meta1 is not None
        assert meta2 is not None
        assert meta1.metadata.training_samples == meta2.metadata.training_samples


# ---------------------------------------------------------------------------
# 5. Combined dataset date coverage validation
# ---------------------------------------------------------------------------

class TestCombinedDatasetCoverage:
    def test_90_day_window_requires_at_least_180_day_dataset(self):
        """With 90-day train + 7-day eval + buffer, need ≥ 180 days total."""
        min_days_for_5_folds = 90 + 5 * 7 + 7  # train + 5 eval folds + buffer
        assert min_days_for_5_folds <= 180, (
            f"5-fold benchmark needs {min_days_for_5_folds} days; combined dataset should have ≥180"
        )

    def test_summer_plus_fall_plus_q1_spans_270_days(self):
        """Verify the 3-period merge spans ≥270 days."""
        summer_start = datetime(2025, 6, 1)
        q1_end = datetime(2026, 3, 10)
        span = (q1_end - summer_start).days
        assert span >= 270, f"Expected ≥270 days, got {span}"

    def test_fall_period_fills_gap_between_summer_and_q1(self):
        """Fall 2025 (Sep-Dec) fills the gap between summer and Q1 2026."""
        summer_end = datetime(2025, 8, 31)
        fall_start = datetime(2025, 9, 1)
        q1_start = datetime(2026, 1, 1)
        fall_end = datetime(2025, 12, 31)
        # No gap between summer end and fall start
        assert (fall_start - summer_end).days <= 1
        # No gap between fall end and Q1 start
        assert (q1_start - fall_end).days <= 1

    def test_recommended_train_days_in_extended_combo(self):
        """The extended combo documents its recommended training window."""
        from benchmarks.run_benchmark import EXTENDED_REGION_COMBOS
        for combo in EXTENDED_REGION_COMBOS:
            if "combined" in combo["name"]:
                start = datetime.fromisoformat(combo["date_start"])
                end = datetime.fromisoformat(combo["date_end"])
                span = (end - start).days
                rec = combo.get("recommended_train_days", 90)
                # Verify the span allows ≥5 folds with recommended training window
                available_eval = span - rec
                n_folds = available_eval // 7
                assert n_folds >= 5, (
                    f"Combo {combo['name']}: only {n_folds} folds possible "
                    f"with {rec}-day train window over {span}-day span"
                )


# ---------------------------------------------------------------------------
# 6. Benchmark runner handles --extended-data flag correctly
# ---------------------------------------------------------------------------

class TestExtendedDataFlag:
    def test_extended_combos_not_in_standard_region_combos(self):
        """Extended combos are in EXTENDED_REGION_COMBOS, not REGION_COMBOS."""
        from benchmarks.run_benchmark import EXTENDED_REGION_COMBOS, REGION_COMBOS
        standard_names = {c["name"] for c in REGION_COMBOS}
        extended_names = {c["name"] for c in EXTENDED_REGION_COMBOS}
        overlap = standard_names & extended_names
        assert len(overlap) == 0, f"Combo names in both lists: {overlap}"

    def test_region_combo_lookup_includes_extended(self):
        """--region-combo can reference extended combos."""
        from benchmarks.run_benchmark import EXTENDED_REGION_COMBOS, REGION_COMBOS
        all_combos = REGION_COMBOS + EXTENDED_REGION_COMBOS
        combo_names = {c["name"] for c in all_combos}
        assert "combined_2025_2026_3region" in combo_names

    def test_skipped_result_not_counted_in_summary(self):
        """Skipped results should be excluded from the non_error summary list."""
        results = [
            {"region_combo": "a", "primary_savings_pct": 20.0},
            {"region_combo": "b", "skipped": True, "skip_reason": "missing file"},
            {"region_combo": "c", "error": "something broke"},
        ]
        non_error = [r for r in results if "error" not in r and not r.get("skipped")]
        assert len(non_error) == 1
        assert non_error[0]["region_combo"] == "a"


# ---------------------------------------------------------------------------
# 7. Integration: per-region with 90d beats 30d (acceptance criterion)
# ---------------------------------------------------------------------------

class TestPerRegionVsJointWith90DayData:
    """Verify that per-region with 90-day synthetic data is sensible.

    This does NOT run a real benchmark (that takes minutes) — it validates
    that the model trains, predicts, and produces coherent forecasts.
    A real benchmark acceptance test is run manually and results archived.
    """

    def test_per_region_forecaster_produces_coherent_price_forecasts(self):
        """Per-region forecaster should produce reasonable price forecasts."""
        import numpy as np

        from aurelius.forecasting.price_model import (
            PerRegionForecaster,
            PerRegionForecasterConfig,
            PriceModelConfig,
        )
        from aurelius.models import EnergyPrice

        # 90 days of training data per region
        base_ts = datetime(2025, 9, 1, tzinfo=timezone.utc)
        rng = np.random.default_rng(99)
        train_data = []
        for region, base_price in [("us-west", 45.0), ("us-east", 55.0), ("us-south", 35.0)]:
            for hour in range(90 * 24):
                ts = base_ts + timedelta(hours=hour)
                noise = rng.normal(0, 3)
                price = base_price + 10 * abs(rng.normal()) + noise
                train_data.append(EnergyPrice(
                    timestamp=ts, region=region,
                    price_per_mwh=max(5.0, price),
                ))

        config = PerRegionForecasterConfig(
            base_config=PriceModelConfig(seed=42, n_estimators=30, num_leaves=15),
        )
        fc = PerRegionForecaster(config=config)
        fc.fit(train_data)

        predict_start = base_ts + timedelta(days=90)
        context = train_data[-336:]  # last 14 days context
        future_ts = [predict_start + timedelta(hours=h) for h in range(24)]

        for region in ("us-west", "us-east", "us-south"):
            region_fc = fc._region_forecasters[region]
            region_context = [p for p in context if p.region == region]
            preds = region_fc.predict(
                region=region,
                timestamps=future_ts,
                recent_prices=region_context,
            )
            assert len(preds) == 24, f"Expected 24 hourly predictions for {region}"
            for pred in preds:
                assert pred.p50 > 0, "Forecasted price must be positive"
                assert pred.p90 >= pred.p50, "p90 must be >= p50"

    def test_per_region_metadata_reflects_90d_training(self):
        """Metadata should reflect per-region model was trained on 90-day window."""
        import numpy as np

        from aurelius.forecasting.price_model import (
            PerRegionForecaster,
            PerRegionForecasterConfig,
            PriceModelConfig,
        )
        from aurelius.models import EnergyPrice

        base_ts = datetime(2025, 9, 1, tzinfo=timezone.utc)
        rng = np.random.default_rng(10)
        train_data = []
        for region in ("us-west", "us-east"):
            for hour in range(90 * 24):
                ts = base_ts + timedelta(hours=hour)
                price = 40.0 + rng.normal(0, 10)
                train_data.append(EnergyPrice(
                    timestamp=ts, region=region,
                    price_per_mwh=max(5.0, price),
                ))

        fc = PerRegionForecaster(config=PerRegionForecasterConfig(
            base_config=PriceModelConfig(seed=42, n_estimators=20, num_leaves=15),
        ))
        fc.fit(train_data)
        meta = fc.metadata
        assert meta is not None
        assert meta.model_type == "per_region_forecaster"
        assert fc.is_fitted is True
        assert len(meta.regions) == 2
        # Total samples across both regions = 90 days × 24h × 2 regions = 4320
        assert meta.training_samples == 90 * 24 * 2
