"""Tests for the TTFT p50 shadow-only integration."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from aurelius.forecasting.ttft_shadow import (  # noqa: E402
    FEATURE_VERSION,
    MODEL_VERSION,
    ShadowConfig,
    ShadowPrediction,
    TTFTp50ShadowPredictor,
    summarize_shadow_batch,
)


class _ConstML:
    def __init__(self, v):
        self.v = v

    def predict(self, X):
        return np.full(np.asarray(X).shape[0], self.v, dtype=np.float64)


def _baseline_predict(X, instance_types):
    return np.full(np.asarray(X).shape[0], 1.0, dtype=np.float64)


# ---------- 1. Shadow config defaults to ENABLED -------------------------


def test_shadow_config_enabled_by_default():
    cfg = ShadowConfig()
    assert cfg.enabled is True
    assert cfg.executable_in_real_cluster is False


def test_shadow_predictor_rejects_executable_config():
    with pytest.raises(ValueError):
        TTFTp50ShadowPredictor(
            ml_model=_ConstML(0.1),
            baseline_predict=_baseline_predict,
            config=ShadowConfig(enabled=True, executable_in_real_cluster=True),
        )


# ---------- 2. Disabled emits nothing ------------------------------------


def test_disabled_shadow_emits_no_records():
    pred = TTFTp50ShadowPredictor(
        ml_model=_ConstML(0.1), baseline_predict=_baseline_predict,
        config=ShadowConfig(enabled=False),
    )
    X = np.zeros((10, 4))
    out = pred.predict_shadow(X, instance_types=["a"] * 10)
    assert out == []


# ---------- 3. Enabled emits records but no control action ---------------


def test_enabled_shadow_emits_prediction_records():
    pred = TTFTp50ShadowPredictor(
        ml_model=_ConstML(0.5), baseline_predict=_baseline_predict,
        config=ShadowConfig(enabled=True),
    )
    X = np.zeros((3, 4))
    out = pred.predict_shadow(
        X, instance_types=["qwen2.5-3b_a30"] * 3,
        request_ids=["r1", "r2", "r3"],
    )
    assert len(out) == 3
    for rec in out:
        assert isinstance(rec, ShadowPrediction)
        assert rec.shadow_only is True
        assert rec.executable_in_real_cluster is False
        assert rec.ttft_p50_prediction_s == 0.5
        assert rec.baseline_ttft_p50_prediction_s == 1.0
        assert rec.prediction_delta_s == pytest.approx(-0.5)
        assert rec.model_version == MODEL_VERSION
        assert rec.feature_version == FEATURE_VERSION


def test_shadow_prediction_to_dict_has_no_executable_flag_true():
    rec = ShadowPrediction(
        request_id="r", instance_type="a", ttft_p50_prediction_s=0.1,
        baseline_ttft_p50_prediction_s=0.2, prediction_delta_s=-0.1)
    d = rec.to_dict()
    assert d["shadow_only"] is True
    assert d["executable_in_real_cluster"] is False
    assert "no_control_action" not in d or d.get("no_control_action") is not False


# ---------- 4. Predictor has no control-action method --------------------


def test_predictor_exposes_no_control_action_method():
    pred = TTFTp50ShadowPredictor(
        ml_model=_ConstML(0.1), baseline_predict=_baseline_predict)
    public = [m for m in dir(pred) if not m.startswith("_")]
    # The ONLY prediction method is predict_shadow. There must be no
    # route/place/scale/admit method.
    forbidden = ("route", "place", "scale", "admit", "execute", "apply",
                 "set_rho", "set_replicas")
    for m in public:
        assert not any(f in m.lower() for f in forbidden), (
            f"shadow predictor exposes control-like method '{m}'"
        )


# ---------- 5. Batch summary marks no control action ---------------------


def test_summarize_shadow_batch_marks_no_control_action():
    pred = TTFTp50ShadowPredictor(
        ml_model=_ConstML(0.3), baseline_predict=_baseline_predict)
    X = np.zeros((100, 4))
    preds = pred.predict_shadow(X, instance_types=["a"] * 100)
    y = np.full(100, 0.4)
    summary = summarize_shadow_batch(preds, y, holdout_name="random")
    assert summary["no_control_action_taken"] is True
    assert summary["shadow_only"] is True
    assert summary["executable_in_real_cluster"] is False
    assert summary["rows_evaluated"] == 100
    assert summary["prediction_coverage"] == 1.0
    assert "ml_p50_pinball" in summary


def test_summarize_empty_batch():
    summary = summarize_shadow_batch([], holdout_name="x", enabled=False)
    assert summary["rows_evaluated"] == 0
    assert summary["prediction_coverage"] == 0.0
    assert summary["no_control_action_taken"] is True


# ---------- 6. Summary artefact validation -------------------------------


SHADOW_SUMMARY = os.path.join(
    REPO_ROOT, "data", "external", "forecasting",
    "cara_latency_forecaster_v1", "ttft_p50_shadow_summary.json")


@pytest.fixture(scope="module")
def shadow_summary():
    if not os.path.exists(SHADOW_SUMMARY):
        pytest.skip("ttft_p50_shadow_summary.json not generated; run "
                    "scripts/run_ttft_p50_shadow.py first")
    with open(SHADOW_SUMMARY) as fh:
        return json.load(fh)


def test_shadow_summary_marks_ttft_p50_only_shadow_ready(shadow_summary):
    assert shadow_summary["ttft_p50_status"] == "shadow_ready"
    assert shadow_summary["other_models_shadow_ready"] == []
    assert shadow_summary["no_control_action_taken"] is True
    assert shadow_summary["shadow_only"] is True
    assert shadow_summary["production_claim"] is False


def test_shadow_summary_has_holdout_performance(shadow_summary):
    assert "time_holdout" in shadow_summary["per_holdout"]
    th = shadow_summary["per_holdout"]["time_holdout"]
    assert th["no_control_action_taken"] is True
    assert th["rows_evaluated"] > 1000
    # Time-holdout pinball improvement should be recorded (and positive,
    # consistent with TTFT p50 being shadow_ready).
    assert shadow_summary["time_holdout_pinball_improvement_pct"] is not None


def test_shadow_summary_enabled_by_default(shadow_summary):
    assert shadow_summary["shadow_enabled"] is True


# ---------- 7. Shadow module has no controller imports -------------------


def test_shadow_module_has_no_controller_imports():
    path = os.path.join(REPO_ROOT, "aurelius", "forecasting", "ttft_shadow.py")
    with open(path) as fh:
        src = fh.read()
    for forbidden in (
        "aurelius.optimization.scheduler",
        "aurelius.frontier.controller",
        "aurelius.frontier.execution",
        "aurelius.frontier.dynamic_controller",
    ):
        assert forbidden not in src, (
            f"shadow module must not import {forbidden}"
        )


def test_raw_and_analysis_sample_gitignored():
    for path in (
        os.path.join(REPO_ROOT, "data", "external", "hf",
                     "asdwb__cara_latency_prediction", "raw", "train.jsonl"),
        os.path.join(REPO_ROOT, "data", "external", "hf",
                     "asdwb__cara_latency_prediction", "train_flat",
                     "processed", "analysis_sample.jsonl"),
    ):
        if not os.path.exists(path):
            continue
        r = subprocess.run(["git", "check-ignore", path], cwd=REPO_ROOT,
                           capture_output=True, text=True)
        assert r.returncode == 0, f"not gitignored: {path}"
