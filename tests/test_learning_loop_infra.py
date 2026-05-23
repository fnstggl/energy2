"""Tests for the productionized learning-loop infrastructure:

- FileLock (overlapping-run prevention)
- model registry + rollback (DB)
- run_model_update (honest candidate-vs-active promotion)
- learning-run lifecycle

All DB tests use SQLite; model training uses small synthetic price data.
"""
from __future__ import annotations

import tempfile
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from aurelius.database import TimeSeriesStore
from aurelius.learning.locking import FileLock, LockNotAcquiredError
from aurelius.learning.promotion import dataset_hash, run_model_update
from aurelius.storage import LocalArtifactStore

# ---------------------------------------------------------------------------
# FileLock
# ---------------------------------------------------------------------------

class TestFileLock:
    def test_second_acquire_blocks(self, tmp_path):
        lock_path = str(tmp_path / "x.lock")
        a = FileLock(lock_path)
        a.acquire()
        b = FileLock(lock_path)
        with pytest.raises(LockNotAcquiredError):
            b.acquire()
        a.release()

    def test_reacquire_after_release(self, tmp_path):
        lock_path = str(tmp_path / "x.lock")
        a = FileLock(lock_path)
        a.acquire()
        a.release()
        b = FileLock(lock_path)
        b.acquire()  # should not raise
        b.release()

    def test_context_manager(self, tmp_path):
        lock_path = str(tmp_path / "x.lock")
        with FileLock(lock_path):
            with pytest.raises(LockNotAcquiredError):
                FileLock(lock_path).acquire()


# ---------------------------------------------------------------------------
# Model registry + rollback
# ---------------------------------------------------------------------------

@pytest.fixture
def store():
    s = TimeSeriesStore("sqlite:///:memory:")
    yield s
    s.close()


class TestModelRegistry:
    def test_register_and_get_none_active(self, store):
        store.register_model("m1", "v1", "file:///m1", scope="acme", pilot_id="p1")
        # candidate is not active yet
        assert store.get_active_model("price", "acme", "p1") is None

    def test_promote_sets_active(self, store):
        store.register_model("m1", "v1", "file:///m1", scope="acme", pilot_id="p1")
        assert store.promote_model("m1") is True
        active = store.get_active_model("price", "acme", "p1")
        assert active["model_id"] == "m1"
        assert active["status"] == "active"

    def test_promote_archives_prior_active(self, store):
        store.register_model("m1", "v1", "file:///m1", scope="acme", pilot_id="p1")
        store.promote_model("m1")
        store.register_model("m2", "v2", "file:///m2", scope="acme", pilot_id="p1", parent_model_id="m1")
        store.promote_model("m2")
        assert store.get_active_model("price", "acme", "p1")["model_id"] == "m2"
        assert store.get_model("m1")["status"] == "archived"

    def test_rollback_restores_previous(self, store):
        store.register_model("m1", "v1", "file:///m1", scope="acme", pilot_id="p1")
        store.promote_model("m1")
        store.register_model("m2", "v2", "file:///m2", scope="acme", pilot_id="p1")
        store.promote_model("m2")
        restored = store.rollback_active("price", "acme", "p1")
        assert restored["model_id"] == "m1"
        assert store.get_model("m2")["status"] == "rolled_back"
        assert store.get_active_model("price", "acme", "p1")["model_id"] == "m1"

    def test_rollback_none_when_no_previous(self, store):
        store.register_model("m1", "v1", "file:///m1", scope="acme", pilot_id="p1")
        store.promote_model("m1")
        assert store.rollback_active("price", "acme", "p1") is None

    def test_customer_isolation(self, store):
        store.register_model("a1", "v1", "file:///a1", scope="acme", pilot_id="p1")
        store.promote_model("a1")
        store.register_model("g1", "v1", "file:///g1", scope="globex", pilot_id="p1")
        store.promote_model("g1")
        assert store.get_active_model("price", "acme", "p1")["model_id"] == "a1"
        assert store.get_active_model("price", "globex", "p1")["model_id"] == "g1"

    def test_eval_metrics_roundtrip(self, store):
        store.register_model("m1", "v1", "file:///m1", scope="acme", pilot_id="p1",
                             eval_metrics={"mae": 5.0, "mape": 0.1})
        assert store.get_model("m1")["eval_metrics"] == {"mae": 5.0, "mape": 0.1}

    def test_decisions_logged(self, store):
        store.record_promotion_decision("promote", scope="acme", pilot_id="p1",
                                        model_id="m1", primary_metric="mae",
                                        candidate_value=4.0, active_value=5.0, reason="better")
        decs = store.get_promotion_decisions(scope="acme")
        assert len(decs) == 1
        assert decs[0]["decision"] == "promote"

    def test_disabled_store_safe(self):
        s = TimeSeriesStore(url="")
        assert s.register_model("m", "v", "file:///m") is False
        assert s.get_active_model() is None
        assert s.promote_model("m") is False
        assert s.rollback_active() is None


class TestLearningRunLifecycle:
    def test_start_and_finish(self, store):
        assert store.start_learning_run("run1", scope="acme", pilot_id="p1") is True
        assert store.finish_learning_run("run1", state="completed", summary={"ok": 1}) is True
        assert store.row_counts()["learning_runs"] == 1

    def test_disabled_returns_false(self):
        s = TimeSeriesStore(url="")
        assert s.start_learning_run("r") is False
        assert s.finish_learning_run("r") is False


# ---------------------------------------------------------------------------
# Honest candidate-vs-active promotion
# ---------------------------------------------------------------------------

def _synth_prices(days: int = 50, regions=("us-west", "us-east")) -> pd.DataFrame:
    """Deterministic synthetic hourly prices with a daily cycle + region offset."""
    rng = np.random.default_rng(0)
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = []
    for h in range(days * 24):
        ts = t0 + timedelta(hours=h)
        hour = ts.hour
        for i, r in enumerate(regions):
            base = 40 + 15 * np.sin(2 * np.pi * hour / 24) + i * 5
            rows.append({"timestamp": ts, "region": r,
                         "price_per_mwh": float(base + rng.normal(0, 3))})
    return pd.DataFrame(rows)


class TestDatasetHash:
    def test_deterministic(self):
        df = _synth_prices(5)
        assert dataset_hash(df) == dataset_hash(df.sample(frac=1, random_state=1))

    def test_changes_with_content(self):
        df = _synth_prices(5)
        df2 = df.copy()
        df2.loc[0, "price_per_mwh"] += 1.0
        assert dataset_hash(df) != dataset_hash(df2)

    def test_empty(self):
        assert dataset_hash(pd.DataFrame()) == "empty"


class TestRunModelUpdate:
    @pytest.fixture(autouse=True)
    def _quiet(self):
        import warnings
        warnings.filterwarnings("ignore")

    def _cls_cfg(self):
        from aurelius.forecasting.price_model import PriceModelConfig, PriceQuantileForecaster
        return PriceQuantileForecaster, PriceModelConfig(seed=42, n_estimators=40, num_leaves=15)

    def test_skipped_on_small_data(self, store):
        art = LocalArtifactStore(base_dir=tempfile.mkdtemp())
        cls, cfg = self._cls_cfg()
        out = run_model_update(_synth_prices(2), ["us-west", "us-east"], cls, cfg,
                               store, art, eval_days=7)
        assert out["status"] == "skipped"

    def test_first_run_promotes_and_persists(self, store):
        art = LocalArtifactStore(base_dir=tempfile.mkdtemp())
        cls, cfg = self._cls_cfg()
        out = run_model_update(_synth_prices(50), ["us-west", "us-east"], cls, cfg,
                               store, art, eval_days=7, scope="acme", pilot_id="p1", run_id="r1")
        assert out["status"] == "ok"
        assert out["promoted"] is True
        assert out["reason"] == "no_active_model"
        # Persisted: registry has an active model + a promote decision
        assert store.get_active_model("price", "acme", "p1")["model_id"] == out["model_id"]
        assert art.exists(out["artifact_uri"])
        assert store.get_promotion_decisions(scope="acme")[0]["decision"] == "promote"

    def test_identical_candidate_is_rejected(self, store):
        """Loads the persisted active model and rejects a non-improving candidate."""
        art = LocalArtifactStore(base_dir=tempfile.mkdtemp())
        cls, cfg = self._cls_cfg()
        df = _synth_prices(50)
        run_model_update(df, ["us-west", "us-east"], cls, cfg, store, art,
                         eval_days=7, scope="acme", pilot_id="p1", run_id="r1")
        out2 = run_model_update(df, ["us-west", "us-east"], cls, cfg, store, art,
                                eval_days=7, scope="acme", pilot_id="p1", run_id="r2")
        assert out2["status"] == "ok"
        assert out2["promoted"] is False
        assert out2["active_value"] is not None  # active model WAS loaded + evaluated

    def test_dry_run_persists_nothing(self, store):
        art = LocalArtifactStore(base_dir=tempfile.mkdtemp())
        cls, cfg = self._cls_cfg()
        out = run_model_update(_synth_prices(50), ["us-west", "us-east"], cls, cfg,
                               store, art, eval_days=7, scope="acme", pilot_id="p1", dry_run=True)
        assert out["status"] == "ok"
        assert store.get_active_model("price", "acme", "p1") is None
        assert store.row_counts()["model_registry"] == 0

    def test_works_without_store(self):
        """No DATABASE_URL: still trains/evaluates, just no registry persistence."""
        art = LocalArtifactStore(base_dir=tempfile.mkdtemp())
        cls, cfg = self._cls_cfg()
        disabled = TimeSeriesStore(url="")
        out = run_model_update(_synth_prices(50), ["us-west", "us-east"], cls, cfg,
                               disabled, art, eval_days=7)
        assert out["status"] == "ok"
        assert out["promoted"] is True  # no active model anywhere
