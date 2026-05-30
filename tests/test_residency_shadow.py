"""Tests for recommendation-only residency shadow logging.

The binding properties under test: shadow decisions are **recommendation-only**
(``executed=False``, no mutation path), they NEVER substitute the requested
model/adapter, and insufficient telemetry yields ``insufficient_telemetry``
(the recommender does not guess).
"""

from __future__ import annotations

import os

import pytest

from aurelius.residency import ingest, shadow
from aurelius.residency.models import RequestResidencyObservation
from aurelius.residency.shadow import (
    ResidencyMutationError,
    ResidencyShadowDecision,
    ResidencyShadowLog,
    ShadowRecommender,
    ShadowRecommenderConfig,
)

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "residency")
REQUESTS = os.path.join(FIX, "requests.jsonl")


# ---------------------------------------------------------------------------
# Recommendation-only invariants
# ---------------------------------------------------------------------------

def test_decision_defaults_to_not_executed():
    d = ResidencyShadowDecision(
        timestamp=1.0, workload_id="w", model_id="m", current_location="pod-a",
        recommended_action="no_op", reason="r")
    assert d.executed is False


def test_decision_rejects_executed_true():
    with pytest.raises(ResidencyMutationError):
        ResidencyShadowDecision(
            timestamp=1.0, workload_id="w", model_id="m", current_location="pod-a",
            recommended_action="prewarm", reason="r", executed=True)


def test_log_refuses_executed_decision():
    log = ResidencyShadowLog()
    d = ResidencyShadowDecision(
        timestamp=1.0, workload_id="w", model_id="m", current_location=None,
        recommended_action="prewarm", reason="r")
    object.__setattr__(d, "executed", True)  # force the invalid state
    with pytest.raises(ResidencyMutationError):
        log.record(d)


def test_module_never_allows_mutation():
    assert shadow.MUTATION_ALLOWED is False


def test_decision_rejects_unknown_action():
    with pytest.raises(ValueError):
        ResidencyShadowDecision(
            timestamp=1.0, workload_id="w", model_id="m", current_location=None,
            recommended_action="delete_cluster", reason="r")


# ---------------------------------------------------------------------------
# Recommendation logic
# ---------------------------------------------------------------------------

def test_insufficient_telemetry_when_residency_unknown():
    obs = RequestResidencyObservation(request_id="r", timestamp=1.0, model_id="m",
                                      source="t")  # unknown residency
    d = ShadowRecommender().recommend(obs)
    assert d.recommended_action == "insufficient_telemetry"
    assert d.expected_cold_start_saved_s is None


def test_cold_start_recommends_prewarm():
    obs = RequestResidencyObservation(
        request_id="r", timestamp=1.0, model_id="m", source="t",
        model_loaded_before_request=False, model_load_latency_s=22.0,
        e2e_latency_s=25.0)
    d = ShadowRecommender(ShadowRecommenderConfig(slo_s=5.0)).recommend(obs)
    assert d.recommended_action == "prewarm"
    assert d.expected_cold_start_saved_s == 22.0
    assert d.expected_sla_risk_delta == -1.0  # would have met SLO without the load


def test_warm_hit_recommends_preserve_affinity():
    obs = RequestResidencyObservation(
        request_id="r", timestamp=1.0, model_id="m", source="t",
        model_loaded_before_request=True)
    d = ShadowRecommender().recommend(obs)
    assert d.recommended_action == "preserve_affinity"


def test_cheap_cold_start_is_no_op():
    obs = RequestResidencyObservation(
        request_id="r", timestamp=1.0, model_id="m", source="t",
        model_loaded_before_request=False, model_load_latency_s=1.0)
    d = ShadowRecommender(ShadowRecommenderConfig(min_cold_start_s_to_prewarm=5.0)
                          ).recommend(obs)
    assert d.recommended_action == "no_op"


def test_decayed_popularity_flags_evict_candidate():
    obs = RequestResidencyObservation(
        request_id="r", timestamp=1.0, model_id="m", source="t",
        model_loaded_before_request=True)
    d = ShadowRecommender().recommend(obs, popularity_decayed=True)
    assert d.recommended_action == "evict_candidate"


def test_no_substitution_decision_never_changes_requested_model():
    obs = RequestResidencyObservation(
        request_id="r", timestamp=1.0, model_id="requested-model", source="t",
        adapter_id="requested-adapter", model_loaded_before_request=False,
        model_load_latency_s=30.0)
    d = ShadowRecommender().recommend(obs)
    # the recommendation acts on where/when, NEVER on which model is served
    assert d.model_id == "requested-model"
    assert d.adapter_id == "requested-adapter"
    # there is no field by which a substitute model could be expressed
    assert "substitute" not in d.to_dict()


def test_confidence_capped_under_weak_linkage():
    obs = RequestResidencyObservation(
        request_id="r", timestamp=1.0, model_id="m", source="t", confidence="high",
        model_loaded_before_request=False, model_load_latency_s=30.0)
    weak = ShadowRecommender().recommend(obs, linkage_quality="time_join")
    strong = ShadowRecommender().recommend(obs, linkage_quality="container_join")
    assert weak.confidence == "low"        # unattributed → capped
    assert strong.confidence == "high"


# ---------------------------------------------------------------------------
# Batch + logging
# ---------------------------------------------------------------------------

def test_recommend_all_over_fixtures_logs_only():
    obs = ingest.import_observations(REQUESTS).records
    log = ResidencyShadowLog()
    cfg = ShadowRecommenderConfig(slo_s=5.0, gpu_hour_cost=2.5)
    decisions = shadow.recommend_all(obs, config=cfg,
                                     linkage_quality="container_join", log=log)
    assert len(decisions) == 7
    summary = log.summary()
    assert summary["all_recommendation_only"] is True
    assert summary["action_counts"]["prewarm"] == 3       # req-2,6,7
    assert summary["action_counts"]["preserve_affinity"] == 3  # req-1,4,5
    assert summary["action_counts"]["no_op"] == 1         # req-3 (cheap adapter)
    assert summary["total_expected_cold_start_saved_s"] == 92.5


def test_log_writes_jsonl(tmp_path):
    path = tmp_path / "shadow.jsonl"
    log = ResidencyShadowLog(str(path))
    obs = RequestResidencyObservation(
        request_id="r", timestamp=1.0, model_id="m", source="t",
        model_loaded_before_request=True)
    log.record(ShadowRecommender().recommend(obs))
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 1
    rt = ResidencyShadowDecision.from_json(lines[0])
    assert rt.recommended_action == "preserve_affinity" and rt.executed is False
