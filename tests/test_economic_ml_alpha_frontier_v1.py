"""Economic ML Alpha Frontier Refresh v1 — audit-integrity tests.

Proves: the cross-dataset (by_dataset) experiment matrix runs for SwissAI +
Mooncake; label compatibility is recorded; simulator_prior-only targets cannot
become headline (Huawei cold-start is calibration-only, never GPU ML, never
shadow_ready); deterministic cost targets remain diagnostic_only; autoscaling/
queue is proxy-classified; no oracle/FIFO headline; no production behavior
changed; no production-savings claim; the cache verdict is honest (NOT
shadow_ready, because cross-dataset transfer did not beat the target's own
baseline both ways).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ART = REPO_ROOT / "data" / "external" / "forecasting" / "economic_ml_alpha_frontier_v1"


@pytest.fixture(scope="module")
def summary():
    return json.loads((ART / "summary.json").read_text())


@pytest.fixture(scope="module")
def eval_():
    return json.loads((ART / "economic_alpha_eval.json").read_text())


@pytest.fixture(scope="module")
def trained():
    return json.loads((ART / "trained_models.json").read_text())


@pytest.fixture(scope="module")
def catalog():
    return json.loads((ART / "target_catalog.json").read_text())


def test_cross_dataset_matrix_runs(trained):
    exp = trained["cache_reuse_experiments"]
    for key in ("swissai_only", "mooncake_only", "swissai_to_mooncake", "mooncake_to_swissai"):
        assert key in exp, f"missing experiment {key}"
    assert trained["n_swissai_rows"] > 1000 and trained["n_mooncake_rows"] > 1000


def test_label_compatibility_recorded(eval_):
    ans = eval_["answer"]
    lc = ans["label_compatibility"].lower()
    assert "measured" in lc and "derived" in lc and "proxy" in lc


def test_cache_verdict_is_honest_not_shadow_ready(summary):
    v = summary["cache_reuse_verdict"]
    assert v["status"] in ("single_dataset_promising_only",
                           "proxy_promising_needs_pilot_validation", "diagnostic_only")
    assert v["status"] != "shadow_ready_for_integration_review", \
        "cannot be shadow_ready: second source is a DERIVED proxy"
    # the rigorous test must be recorded
    assert "transfer_vs_target_own_baseline" in v


def test_rigorous_transfer_test_used(summary):
    v = summary["cache_reuse_verdict"]
    t = v["transfer_vs_target_own_baseline"]
    # both-direction flag exists and the verdict respects it
    assert "both_directions_beat_target_own_baseline" in t
    if not t["both_directions_beat_target_own_baseline"]:
        assert v["status"] in ("single_dataset_promising_only", "diagnostic_only")


def test_cold_start_is_calibration_prior_not_gpu_ml(summary, catalog, trained):
    cs = summary["cold_start_verdict"]
    assert cs["status"] == "simulator_prior_calibrated"
    assert cs["is_gpu_model_load"] is False
    assert cs["calibration_only"] is True
    # cold-start cannot appear as a shadow-ready / headline target
    assert catalog["cold_start_cost"]["is_gpu_model_load"] is False
    assert trained["cold_start_prior"]["gpu_llm_ml_training"].startswith("BLOCKED")


def test_simulator_prior_only_cannot_be_headline(eval_):
    # economic alpha eval headline is the cache question + KPI, not cold-start/autoscaling
    assert eval_["binding_question"].startswith("Does cache_reuse_pct")
    assert eval_["cold_start_prior_status"] == "simulator_prior_calibrated"
    assert eval_["autoscaling_queue_proxy_status"] in ("proxy_trained", "insufficient", "absent")


def test_autoscaling_queue_is_proxy(summary, catalog):
    aq = summary["autoscaling_queue_verdict"]
    assert aq["label_class"] == "proxy"
    assert aq["has_measured_serving_autoscaling"] is False
    assert catalog["autoscaling_queue_risk"]["label_class"] == "proxy"


def test_deterministic_cost_target_stays_diagnostic(catalog):
    carried = catalog["carried_from_v1_unchanged"]
    assert carried["estimated_gpu_cost_usd_DETERMINISTIC"] == "diagnostic_only_deterministic_formula"


def test_no_oracle_or_fifo_headline(eval_):
    assert eval_["uses_oracle_as_headline"] is False
    assert eval_["uses_fifo_as_headline"] is False


def test_no_production_claim_or_behavior_change(summary, eval_):
    assert summary["no_production_behavior_change"] is True
    assert summary["production_claim"] is False
    assert summary["real_execution"] is False
    assert eval_["shadow_only"] is True
    # external claim guardrails forbid the dangerous claims
    cannot = " ".join(summary["external_claim_guardrails"]["cannot_claim"]).lower()
    assert "production savings" in cannot
    assert "gpu model-load" in cannot or "faas != gpu" in cannot


def test_not_more_production_plausible(summary):
    # since cross-dataset failed, the model is NOT more production-plausible
    assert summary["becomes_more_production_plausible"] is False
