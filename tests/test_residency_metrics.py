"""Tests for derived residency / cold-start metrics.

The binding property under test: every metric handles missing data *honestly* —
unknown residency flags are excluded from the denominator (never counted as a
miss), missing latencies are never treated as 0.0, and an empty denominator
yields ``value=None`` with ``note="insufficient_telemetry"`` (never a
misleading 0.0).
"""

from __future__ import annotations

import os

from aurelius.residency import ingest, metrics
from aurelius.residency.models import RequestResidencyObservation

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "residency")
EVENTS = os.path.join(FIX, "events.jsonl")
SNAPSHOTS = os.path.join(FIX, "snapshots.jsonl")
REQUESTS = os.path.join(FIX, "requests.jsonl")


def _load():
    return (ingest.import_observations(REQUESTS).records,
            ingest.import_events(EVENTS).records,
            ingest.import_snapshots(SNAPSHOTS).records)


# ---------------------------------------------------------------------------
# Hit / miss / cold-start rates
# ---------------------------------------------------------------------------

def test_model_residency_hit_rate_excludes_unknowns():
    obs, _, _ = _load()
    m = metrics.model_residency_hit_rate(obs)
    # 7 requests all have a known model flag; 4 are hits (req-1,3,4,5)
    assert m.denominator == 7
    assert m.numerator == 4
    assert abs(m.value - 4 / 7) < 1e-9


def test_unknown_flag_not_counted_as_miss():
    obs = [
        RequestResidencyObservation(request_id="a", timestamp=1.0, model_id="m",
                                    source="t", model_loaded_before_request=True),
        RequestResidencyObservation(request_id="b", timestamp=2.0, model_id="m",
                                    source="t", model_loaded_before_request=None),
    ]
    m = metrics.model_residency_hit_rate(obs)
    # the unknown row is excluded from the denominator, not counted as a miss
    assert m.denominator == 1 and m.numerator == 1 and m.value == 1.0
    assert m.extra["unknown_excluded"] == 1


def test_adapter_hit_rate_only_over_known_adapter_requests():
    obs, _, _ = _load()
    m = metrics.adapter_residency_hit_rate(obs)
    # req-3 (flag False) + req-4 (flag unknown); only req-3 counts → 0/1
    assert m.denominator == 1 and m.numerator == 0 and m.value == 0.0
    assert m.extra["adapter_flag_unknown"] == 1


def test_cold_start_rate():
    obs, _, _ = _load()
    m = metrics.cold_start_rate(obs)
    assert m.denominator == 7 and m.numerator == 4  # req-2,3,6,7 are cold


def test_metric_insufficient_telemetry_is_none_not_zero():
    obs = [RequestResidencyObservation(request_id="a", timestamp=1.0, model_id="m",
                                       source="t")]  # unknown residency
    m = metrics.model_residency_hit_rate(obs)
    assert m.value is None and m.note == "insufficient_telemetry"


# ---------------------------------------------------------------------------
# Latency distributions
# ---------------------------------------------------------------------------

def test_model_load_latency_percentiles():
    obs, _, _ = _load()
    p = metrics.model_load_latency_percentiles(obs)
    # measured load latencies: req-2 22.4, req-6 30.1, req-7 40.0
    assert p["model_load_latency_p50"].value == 30.1
    assert p["model_load_latency_p99"].value == 40.0
    assert p["model_load_latency_p50"].denominator == 3


def test_adapter_load_latency_percentiles_single_value():
    obs, _, _ = _load()
    p = metrics.adapter_load_latency_percentiles(obs)
    assert p["adapter_load_latency_p50"].value == 4.3
    assert p["adapter_load_latency_p50"].denominator == 1


def test_percentile_helper_ignores_none():
    assert metrics.percentile([None, None], 50) is None
    assert metrics.percentile([1.0, None, 3.0, None, 5.0], 50) == 3.0  # nearest-rank


# ---------------------------------------------------------------------------
# SLA attribution
# ---------------------------------------------------------------------------

def test_cold_start_attributed_sla_violations_requires_slo():
    obs, _, _ = _load()
    no_slo = metrics.cold_start_attributed_sla_violations(obs, slo_s=None)
    assert no_slo.value is None and no_slo.note == "no_slo_provided"
    m = metrics.cold_start_attributed_sla_violations(obs, slo_s=5.0)
    # req-2,3,6,7 violate and would have met SLO without the load → 4
    assert m.value == 4.0
    assert m.extra["total_violations"] == 4


# ---------------------------------------------------------------------------
# Warm-pool cost (snapshot based) + staleness
# ---------------------------------------------------------------------------

def test_warm_pool_gpu_hours_drops_stale_intervals():
    _, _, snaps = _load()
    m = metrics.warm_pool_gpu_hours(snaps, gpu_hour_cost=2.5)
    # gpu-0 (snap-1→snap-2, 100s) + gpu-1 (100s) = 200s; stale interval dropped
    assert abs(m.value - 200 / 3600.0) < 1e-6
    assert m.extra["resident_intervals"] == 2
    assert m.extra["dropped_stale_intervals"] == 1
    assert abs(m.extra["warm_pool_cost"] - (200 / 3600.0) * 2.5) < 1e-3


def test_warm_pool_none_without_snapshots():
    m = metrics.warm_pool_gpu_hours([])
    assert m.value is None and m.note == "insufficient_telemetry"


# ---------------------------------------------------------------------------
# Churn + confidence + missingness
# ---------------------------------------------------------------------------

def test_residency_churn_score():
    _, events, _ = _load()
    m = metrics.residency_churn_score(events)
    # 5 transitions (2 load_end + 1 adapter_load_end + 1 model_evict + 1 adapter_evict)
    assert m.numerator == 5
    assert m.denominator == 2  # distinct models: mistral-7b, llama-3-8b
    assert m.value is not None and m.value > 0


def test_telemetry_confidence_in_unit_interval():
    obs, events, snaps = _load()
    m = metrics.telemetry_confidence(obs, events, snaps,
                                     linkage_quality="container_join")
    assert 0.0 <= m.value <= 1.0
    assert m.extra["known_residency_flag_frac"] == 1.0


def test_missingness_reports_structural_absence():
    obs, events, snaps = _load()
    miss = metrics.missingness(obs, events, snaps)
    # adapter flag missing in 6/7 observations
    assert miss["observations"]["adapter_loaded_before_request"]["missing"] == 6
    assert miss["observations"]["container_id"]["missing"] == 1


def test_compute_all_metrics_keys_present():
    obs, events, snaps = _load()
    out = metrics.compute_all_metrics(obs, events, snaps, slo_s=5.0,
                                      gpu_hour_cost=2.5,
                                      linkage_quality="container_join")
    for key in ("model_residency_hit_rate", "adapter_residency_hit_rate",
                "cold_start_rate", "model_load_latency_p50",
                "model_load_latency_p95", "model_load_latency_p99",
                "adapter_load_latency_p50", "cold_start_attributed_sla_violations",
                "warm_pool_gpu_hours", "residency_churn_score",
                "telemetry_confidence"):
        assert key in out, f"missing metric {key}"
        assert out[key].linkage_quality == "container_join"


def test_empty_inputs_are_honest():
    out = metrics.compute_all_metrics([], [], [])
    assert out["model_residency_hit_rate"].value is None
    assert out["warm_pool_gpu_hours"].value is None
    assert out["residency_churn_score"].value is None
