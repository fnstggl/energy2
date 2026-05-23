"""Tests for aurelius.ml.model_store.

Each test verifies actual save/load correctness, not just that code runs.
Uses real (small) fitted forecasters to verify joblib round-trip integrity.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from aurelius.ml.model_store import ModelStore
from aurelius.models import CarbonIntensity, EnergyPrice

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_store(tmp_path):
    """A ModelStore backed by a temporary directory."""
    return ModelStore(store_root=tmp_path / "model_store")


@pytest.fixture
def fitted_price_forecaster():
    """A small fitted PriceQuantileForecaster for testing."""
    from aurelius.forecasting.price_model import PriceModelConfig, PriceQuantileForecaster
    base = datetime(2024, 1, 1, 0, tzinfo=timezone.utc)
    prices = [
        EnergyPrice(timestamp=base + timedelta(hours=h), region="us-west", price_per_mwh=50.0 + h % 24)
        for h in range(72)
    ]
    config = PriceModelConfig(seed=42, n_estimators=5)
    fc = PriceQuantileForecaster(config)
    fc.fit(prices)
    return fc


@pytest.fixture
def fitted_carbon_forecaster():
    """A small fitted CarbonQuantileForecaster for testing."""
    from aurelius.forecasting.carbon_model import CarbonModelConfig, CarbonQuantileForecaster
    base = datetime(2024, 1, 1, 0, tzinfo=timezone.utc)
    records = [
        CarbonIntensity(
            timestamp=base + timedelta(hours=h),
            region="us-east",
            gco2_per_kwh=300.0 + (h % 24) * 2,
        )
        for h in range(72)
    ]
    config = CarbonModelConfig(seed=42, n_estimators=5)
    fc = CarbonQuantileForecaster(config)
    fc.fit(records)
    return fc


# ---------------------------------------------------------------------------
# ModelStore basic operations
# ---------------------------------------------------------------------------

class TestModelStoreSave:
    def test_save_creates_version_directory(self, tmp_store, fitted_price_forecaster):
        version_id = tmp_store.save(fitted_price_forecaster, model_type="price")
        version_dir = tmp_store.store_root / "price" / version_id
        assert version_dir.exists()
        assert (version_dir / "model.joblib").exists()
        assert (version_dir / "metadata.json").exists()

    def test_save_version_id_format(self, tmp_store, fitted_price_forecaster):
        version_id = tmp_store.save(fitted_price_forecaster, model_type="price")
        assert version_id.startswith("v_")
        # Should be parseable as a UTC timestamp
        ts_str = version_id[2:]
        datetime.strptime(ts_str, "%Y%m%dT%H%M%S")

    def test_save_custom_version_id(self, tmp_store, fitted_price_forecaster):
        custom_id = "v_test_custom"
        result_id = tmp_store.save(
            fitted_price_forecaster, model_type="price", version_id=custom_id
        )
        assert result_id == custom_id
        assert (tmp_store.store_root / "price" / custom_id).exists()

    def test_save_metadata_stored_correctly(self, tmp_store, fitted_price_forecaster):
        meta = {"eval_metrics": {"mape": 0.08}, "holdout_days": 14}
        version_id = tmp_store.save(
            fitted_price_forecaster, model_type="price", metadata=meta
        )
        loaded_meta = tmp_store.load_metadata("price", version_id)
        assert loaded_meta["eval_metrics"]["mape"] == pytest.approx(0.08)
        assert loaded_meta["holdout_days"] == 14
        assert loaded_meta["model_type"] == "price"
        assert loaded_meta["is_active"] is False

    def test_save_refuses_unfitted_model(self, tmp_store):
        from aurelius.forecasting.price_model import PriceQuantileForecaster
        unfitted = PriceQuantileForecaster()
        with pytest.raises(RuntimeError, match="not fitted"):
            tmp_store.save(unfitted, model_type="price")

    def test_save_invalid_model_type_raises(self, tmp_store, fitted_price_forecaster):
        with pytest.raises(ValueError, match="model_type"):
            tmp_store.save(fitted_price_forecaster, model_type="invalid")

    def test_save_duplicate_version_id_raises(self, tmp_store, fitted_price_forecaster):
        custom_id = "v_duplicate_test"
        tmp_store.save(fitted_price_forecaster, model_type="price", version_id=custom_id)
        with pytest.raises(FileExistsError):
            tmp_store.save(fitted_price_forecaster, model_type="price", version_id=custom_id)


class TestModelStorePromote:
    def test_promote_creates_active_symlink(self, tmp_store, fitted_price_forecaster):
        version_id = tmp_store.save(fitted_price_forecaster, model_type="price")
        tmp_store.promote("price", version_id)
        active_link = tmp_store.store_root / "price" / "active"
        assert active_link.exists() or active_link.is_symlink()

    def test_promote_sets_is_active_in_metadata(self, tmp_store, fitted_price_forecaster):
        version_id = tmp_store.save(fitted_price_forecaster, model_type="price")
        tmp_store.promote("price", version_id)
        meta = tmp_store.load_metadata("price", version_id)
        assert meta["is_active"] is True
        assert "promoted_at_utc" in meta

    def test_promote_updates_active_version(self, tmp_store, fitted_price_forecaster):
        v1 = tmp_store.save(fitted_price_forecaster, model_type="price", version_id="v_first")
        tmp_store.promote("price", v1)
        assert tmp_store.get_active_version("price") == v1

        v2 = tmp_store.save(fitted_price_forecaster, model_type="price", version_id="v_second")
        tmp_store.promote("price", v2)
        assert tmp_store.get_active_version("price") == v2

    def test_promote_nonexistent_version_raises(self, tmp_store):
        with pytest.raises(FileNotFoundError):
            tmp_store.promote("price", "v_nonexistent_99999")

    def test_promote_updates_active_link_when_promoting_newer(
        self, tmp_store, fitted_price_forecaster
    ):
        v1 = tmp_store.save(fitted_price_forecaster, model_type="price", version_id="v_old")
        tmp_store.promote("price", v1)
        v2 = tmp_store.save(fitted_price_forecaster, model_type="price", version_id="v_new")
        tmp_store.promote("price", v2)
        assert tmp_store.get_active_version("price") == "v_new"


class TestModelStoreLoad:
    def test_load_active_returns_fitted_forecaster(self, tmp_store, fitted_price_forecaster):
        from aurelius.forecasting.price_model import PriceQuantileForecaster
        version_id = tmp_store.save(fitted_price_forecaster, model_type="price")
        tmp_store.promote("price", version_id)
        loaded = tmp_store.load_active("price", cls=PriceQuantileForecaster)
        assert isinstance(loaded, PriceQuantileForecaster)
        assert loaded.is_fitted

    def test_load_active_forecaster_produces_same_predictions(
        self, tmp_store, fitted_price_forecaster
    ):
        """Round-trip: loaded model should produce identical predictions to original."""
        from aurelius.forecasting.price_model import PriceQuantileForecaster
        base = datetime(2024, 3, 1, 0, tzinfo=timezone.utc)
        timestamps = [base + timedelta(hours=h) for h in range(5)]

        original_preds = fitted_price_forecaster.predict("us-west", timestamps)

        version_id = tmp_store.save(fitted_price_forecaster, model_type="price")
        tmp_store.promote("price", version_id)
        loaded = tmp_store.load_active("price", cls=PriceQuantileForecaster)
        loaded_preds = loaded.predict("us-west", timestamps)

        for orig, loaded_p in zip(original_preds, loaded_preds):
            assert orig.p50 == pytest.approx(loaded_p.p50, rel=1e-6)
            assert orig.p90 == pytest.approx(loaded_p.p90, rel=1e-6)

    def test_load_version_specific(self, tmp_store, fitted_price_forecaster):
        from aurelius.forecasting.price_model import PriceQuantileForecaster
        tmp_store.save(
            fitted_price_forecaster, model_type="price", version_id="v_specific"
        )
        loaded = tmp_store.load_version("price", "v_specific", cls=PriceQuantileForecaster)
        assert isinstance(loaded, PriceQuantileForecaster)
        assert loaded.is_fitted

    def test_load_active_raises_when_no_active(self, tmp_store):
        with pytest.raises(FileNotFoundError, match="No active model"):
            tmp_store.load_active("price")

    def test_load_version_raises_when_missing(self, tmp_store):
        with pytest.raises(FileNotFoundError):
            tmp_store.load_version("price", "v_missing_99999")

    def test_load_active_carbon_model(self, tmp_store, fitted_carbon_forecaster):
        from aurelius.forecasting.carbon_model import CarbonQuantileForecaster
        version_id = tmp_store.save(fitted_carbon_forecaster, model_type="carbon")
        tmp_store.promote("carbon", version_id)
        loaded = tmp_store.load_active("carbon", cls=CarbonQuantileForecaster)
        assert isinstance(loaded, CarbonQuantileForecaster)
        assert loaded.is_fitted

    def test_load_type_mismatch_raises(self, tmp_store, fitted_price_forecaster):
        from aurelius.forecasting.carbon_model import CarbonQuantileForecaster
        version_id = tmp_store.save(fitted_price_forecaster, model_type="price")
        tmp_store.promote("price", version_id)
        with pytest.raises(TypeError, match="expected"):
            tmp_store.load_active("price", cls=CarbonQuantileForecaster)


class TestModelStoreIntrospection:
    def test_list_versions_empty_initially(self, tmp_store):
        assert tmp_store.list_versions("price") == []

    def test_list_versions_sorted(self, tmp_store, fitted_price_forecaster):
        tmp_store.save(fitted_price_forecaster, model_type="price", version_id="v_aaa")
        tmp_store.save(fitted_price_forecaster, model_type="price", version_id="v_zzz")
        tmp_store.save(fitted_price_forecaster, model_type="price", version_id="v_mmm")
        versions = tmp_store.list_versions("price")
        assert versions == sorted(versions)

    def test_has_active_false_initially(self, tmp_store):
        assert tmp_store.has_active("price") is False

    def test_has_active_true_after_promote(self, tmp_store, fitted_price_forecaster):
        version_id = tmp_store.save(fitted_price_forecaster, model_type="price")
        tmp_store.promote("price", version_id)
        assert tmp_store.has_active("price") is True

    def test_get_active_version_none_when_no_active(self, tmp_store):
        assert tmp_store.get_active_version("price") is None

    def test_load_active_metadata_after_promote(self, tmp_store, fitted_price_forecaster):
        meta = {"test_key": "test_value"}
        version_id = tmp_store.save(
            fitted_price_forecaster, model_type="price", metadata=meta
        )
        tmp_store.promote("price", version_id)
        loaded_meta = tmp_store.load_active_metadata("price")
        assert loaded_meta["test_key"] == "test_value"
        assert loaded_meta["is_active"] is True


# ---------------------------------------------------------------------------
# Integration: retrain_forecaster script
# ---------------------------------------------------------------------------

class TestRetrainScript:
    """Test the retraining pipeline end-to-end."""

    def _write_price_csv(self, path: Path, n_records: int = 720) -> None:
        """Write n_records hourly price records. Default 720h = 30 days."""
        import math
        base = datetime(2024, 1, 1, 0, tzinfo=timezone.utc)
        with open(path, "w") as f:
            f.write("timestamp,region,price_per_mwh\n")
            for i in range(n_records):
                ts = base + timedelta(hours=i)
                price = 50 + 20 * math.sin(2 * 3.14159 * i / 24)
                f.write(f"{ts.isoformat()},us-west,{price:.2f}\n")

    def test_retrain_end_to_end_price(self, tmp_path):
        from scripts.retrain_forecaster import main
        csv_path = tmp_path / "prices.csv"
        store_root = tmp_path / "models"
        self._write_price_csv(csv_path)  # default 720 records = 30 days

        # First run: no active model → should train and promote
        exit_code = main([
            "--model-type", "price",
            "--data-csv", str(csv_path),
            "--store-root", str(store_root),
            "--holdout-days", "7",
            "--min-train-days", "10",
            "--seed", "42",
        ])
        assert exit_code == 0

        # Verify model was saved and promoted
        store = ModelStore(store_root=store_root)
        assert store.has_active("price")
        versions = store.list_versions("price")
        assert len(versions) == 1

    def test_retrain_dry_run_does_not_promote(self, tmp_path):
        from scripts.retrain_forecaster import main
        csv_path = tmp_path / "prices.csv"
        store_root = tmp_path / "models"
        self._write_price_csv(csv_path)  # default 720 records = 30 days

        exit_code = main([
            "--model-type", "price",
            "--data-csv", str(csv_path),
            "--store-root", str(store_root),
            "--holdout-days", "7",
            "--min-train-days", "10",
            "--dry-run",
        ])
        assert exit_code == 0

        store = ModelStore(store_root=store_root)
        # Dry-run: no model should be saved
        assert not store.has_active("price")
        assert store.list_versions("price") == []

    def test_retrain_missing_data_returns_1(self, tmp_path):
        from scripts.retrain_forecaster import main
        exit_code = main([
            "--model-type", "price",
            "--data-csv", str(tmp_path / "nonexistent.csv"),
            "--store-root", str(tmp_path / "models"),
        ])
        assert exit_code == 1

    def test_retrain_insufficient_history_returns_1(self, tmp_path):
        from scripts.retrain_forecaster import main
        csv_path = tmp_path / "prices.csv"
        # Only 50 records (~2 days); hold out 7 days > total history → fails split
        self._write_price_csv(csv_path, n_records=50)
        exit_code = main([
            "--model-type", "price",
            "--data-csv", str(csv_path),
            "--store-root", str(tmp_path / "models"),
            "--holdout-days", "14",
            "--min-train-days", "30",
        ])
        assert exit_code == 1

    def test_temporal_split_leakage_invariant(self, tmp_path):
        """Core leakage-free invariant: max(train_ts) < min(holdout_ts)."""
        from scripts.retrain_forecaster import load_price_csv, temporal_split
        csv_path = tmp_path / "prices.csv"
        self._write_price_csv(csv_path, n_records=200)
        records = load_price_csv(csv_path)
        train, holdout, holdout_start = temporal_split(records, holdout_days=7)

        # Hard leakage check
        from datetime import timezone as _tz
        train_max = max(
            r.timestamp.astimezone(_tz.utc) if r.timestamp.tzinfo else r.timestamp.replace(tzinfo=_tz.utc)
            for r in train
        )
        holdout_min = min(
            r.timestamp.astimezone(_tz.utc) if r.timestamp.tzinfo else r.timestamp.replace(tzinfo=_tz.utc)
            for r in holdout
        )
        assert train_max < holdout_min, (
            f"LEAKAGE: train_max={train_max} >= holdout_min={holdout_min}"
        )

    def test_load_price_csv_parses_correctly(self, tmp_path):
        from scripts.retrain_forecaster import load_price_csv
        csv_path = tmp_path / "test.csv"
        csv_path.write_text(
            "timestamp,region,price_per_mwh\n"
            "2024-01-01T00:00:00Z,us-west,45.50\n"
            "2024-01-01T01:00:00Z,us-west,46.00\n"
            "2024-01-01T02:00:00Z,eu-west,55.00\n"
        )
        records = load_price_csv(csv_path)
        assert len(records) == 3
        assert records[0].region == "us-west"
        assert records[0].price_per_mwh == pytest.approx(45.50)
        assert records[2].region == "eu-west"

    def test_load_carbon_csv_parses_correctly(self, tmp_path):
        from scripts.retrain_forecaster import load_carbon_csv
        csv_path = tmp_path / "test.csv"
        csv_path.write_text(
            "timestamp,region,gco2_per_kwh\n"
            "2024-01-01T00:00:00Z,us-west,300.0\n"
            "2024-01-01T01:00:00Z,us-west,310.0\n"
        )
        records = load_carbon_csv(csv_path)
        assert len(records) == 2
        assert records[0].gco2_per_kwh == pytest.approx(300.0)

    def test_load_csv_skips_malformed_rows(self, tmp_path):
        from scripts.retrain_forecaster import load_price_csv
        csv_path = tmp_path / "bad.csv"
        csv_path.write_text(
            "timestamp,region,price_per_mwh\n"
            "2024-01-01T00:00:00Z,us-west,50.00\n"
            "not-a-date,us-west,50.00\n"    # malformed timestamp
            "2024-01-01T02:00:00Z,us-west,notanumber\n"  # malformed price
            "2024-01-01T03:00:00Z,us-west,55.00\n"
        )
        records = load_price_csv(csv_path)
        assert len(records) == 2  # only valid rows

    def test_second_run_skips_when_no_improvement(self, tmp_path):
        """When a model is already active and no improvement, exit code 2."""
        from scripts.retrain_forecaster import main
        csv_path = tmp_path / "prices.csv"
        store_root = tmp_path / "models"
        self._write_price_csv(csv_path, n_records=720)

        # First run: no active model → promotes
        exit_code_1 = main([
            "--model-type", "price",
            "--data-csv", str(csv_path),
            "--store-root", str(store_root),
            "--holdout-days", "7",
            "--min-train-days", "10",
            "--seed", "42",
            # Require 50% improvement (unreachable with same data)
            "--min-improvement", "50.0",
        ])
        # First run always promotes (no existing active)
        assert exit_code_1 == 0

        # Second run with same data and impossible improvement threshold → no promotion
        exit_code_2 = main([
            "--model-type", "price",
            "--data-csv", str(csv_path),
            "--store-root", str(store_root),
            "--holdout-days", "7",
            "--min-train-days", "10",
            "--seed", "42",
            "--min-improvement", "50.0",
        ])
        # Expect exit code 2: evaluated but not promoted
        assert exit_code_2 == 2
