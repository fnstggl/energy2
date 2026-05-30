"""Tests for residency telemetry models + ingestion adapters + linkage.

Covers: model construction + null handling, generic JSONL/CSV import + schema
validation, the vLLM adapter's honesty (no invented load events), and the
Kubernetes/Prometheus/DCGM linkage classifier (no fake joins).
"""

from __future__ import annotations

import os

import pytest

from aurelius.residency import ingest, linkage
from aurelius.residency.models import (
    ModelResidencyEvent,
    ModelResidencySnapshot,
    RequestResidencyObservation,
    ResidencySchemaError,
    parse_timestamp,
)

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "residency")
EVENTS = os.path.join(FIX, "events.jsonl")
SNAPSHOTS = os.path.join(FIX, "snapshots.jsonl")
REQUESTS = os.path.join(FIX, "requests.jsonl")


# ---------------------------------------------------------------------------
# Models + null handling
# ---------------------------------------------------------------------------

def test_parse_timestamp_handles_iso_epoch_ms_and_null():
    assert parse_timestamp(None) is None
    assert parse_timestamp("") is None
    assert parse_timestamp("NULL") is None
    assert parse_timestamp(1780000000) == 1780000000.0
    # epoch ms auto-detected and divided to seconds
    assert parse_timestamp(1780000000000) == 1780000000.0
    iso = parse_timestamp("2026-05-30T12:00:00Z")
    assert iso is not None and iso > 1_700_000_000


def test_missing_data_is_not_zero():
    obs = RequestResidencyObservation(
        request_id="r", timestamp=1.0, model_id="m", source="t")
    # unknown residency must be None (not False), latency None (not 0.0)
    assert obs.model_loaded_before_request is None
    assert obs.model_load_latency_s is None
    assert obs.is_cold_start is None  # cannot classify → unknown
    assert obs.total_load_latency_s is None


def test_observation_cold_start_logic():
    warm = RequestResidencyObservation(
        request_id="r1", timestamp=1.0, model_id="m", source="t",
        model_loaded_before_request=True)
    assert warm.is_cold_start is False
    cold = RequestResidencyObservation(
        request_id="r2", timestamp=1.0, model_id="m", source="t",
        model_loaded_before_request=False, model_load_latency_s=20.0)
    assert cold.is_cold_start is True
    assert cold.total_load_latency_s == 20.0
    # warm base but cold adapter is still a cold start
    cold_adapter = RequestResidencyObservation(
        request_id="r3", timestamp=1.0, model_id="m", source="t", adapter_id="a",
        model_loaded_before_request=True, adapter_loaded_before_request=False,
        adapter_load_latency_s=4.0)
    assert cold_adapter.is_cold_start is True
    assert cold_adapter.total_load_latency_s == 4.0


def test_event_rejects_unknown_event_type():
    with pytest.raises(ResidencySchemaError):
        ModelResidencyEvent(timestamp=1.0, model_id="m",
                            event_type="frobnicate", source="t")


def test_snapshot_residency_unknown_vs_empty():
    unknown = ModelResidencySnapshot(timestamp=1.0, source="t")
    assert unknown.loaded_model_ids is None      # not reported = unknown
    assert unknown.has_residency is False
    empty = ModelResidencySnapshot.from_dict(
        {"timestamp": 1.0, "loaded_model_ids": [], "source": "t"})
    assert empty.loaded_model_ids == ()          # reported empty = known-empty
    assert empty.has_residency is False


def test_event_round_trip_dict():
    e = ModelResidencyEvent(timestamp=5.0, model_id="m", event_type="model_load_end",
                            source="t", duration_s=22.4)
    d = e.to_dict()
    e2 = ModelResidencyEvent.from_dict(d)
    assert e2.duration_s == 22.4 and e2.event_type == "model_load_end"


# ---------------------------------------------------------------------------
# Generic JSONL/CSV import + schema validation
# ---------------------------------------------------------------------------

def test_import_events_jsonl():
    res = ingest.import_events(EVENTS)
    assert res.n_rows == 10 and res.n_valid == 10 and res.n_errors == 0
    types = {e.event_type for e in res.records}
    assert "model_load_end" in types and "model_evict" in types
    assert res.sources == {"pilot_export": 10}


def test_import_observations_accepts_contract_aliases():
    res = ingest.import_observations(REQUESTS)
    assert res.n_valid == 7
    by_id = {o.request_id: o for o in res.records}
    # req-1 used contract names TTFT / e2e_latency / queue_wait
    assert by_id["req-1"].ttft_s == 0.31
    assert by_id["req-1"].e2e_latency_s == 2.1
    assert by_id["req-1"].queue_wait_s == 0.05
    # req-3 used lora_id alias for adapter_id
    assert by_id["req-3"].adapter_id == "lora-finance"
    # req-6 carries no join keys (missing GPU/node linkage scenario)
    assert by_id["req-6"].has_join_keys is False
    # req-4 adapter flag is unknown (None), NOT False
    assert by_id["req-4"].adapter_loaded_before_request is None


def test_import_snapshots_jsonl():
    res = ingest.import_snapshots(SNAPSHOTS)
    assert res.n_valid == 6
    stale = min(res.records, key=lambda s: s.timestamp)
    assert stale.loaded_model_ids == ("old-model-v1",)


def test_coverage_reports_missingness_honestly():
    res = ingest.import_observations(REQUESTS)
    cov = res.field_coverage
    # adapter flag present in only 1/7 (req-3); the rest are unknown
    assert cov["adapter_loaded_before_request"]["present"] == 1
    assert cov["container_id"]["present"] == 6  # req-6 has none
    assert cov["model_loaded_before_request"]["coverage"] == 1.0


def test_import_raises_on_missing_required_field(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"timestamp": 1.0, "event_type": "model_load_end", "source": "t"}\n')
    res = ingest.import_events(str(bad))
    # missing model_id → row error, not a zero-filled record
    assert res.n_valid == 0 and res.n_errors == 1
    assert "model_id" in res.errors[0]


def test_import_csv_format(tmp_path):
    csv_path = tmp_path / "obs.csv"
    csv_path.write_text(
        "request_id,timestamp,model_id,model_loaded_before_request,source\n"
        "r1,1780000000,llama,true,csv_test\n"
        "r2,1780000001,llama,false,csv_test\n")
    res = ingest.import_observations(str(csv_path))
    assert res.n_valid == 2
    flags = {o.request_id: o.model_loaded_before_request for o in res.records}
    assert flags == {"r1": True, "r2": False}


def test_default_source_when_row_omits_it(tmp_path):
    p = tmp_path / "ev.jsonl"
    p.write_text('{"timestamp": 1.0, "model_id": "m", "event_type": "model_evict"}\n')
    res = ingest.import_events(str(p), source="my_pilot")
    assert res.records[0].source == "my_pilot"


# ---------------------------------------------------------------------------
# vLLM adapter — honest, emits no invented load events
# ---------------------------------------------------------------------------

def test_vllm_adapter_emits_no_load_events_and_unknown_residency():
    metrics_dict = {
        "service_id": "llama-3-8b",
        "prefix_cache_hit_rate": 0.82,
        "ttft_p50_ms": 310.0,
        "p50_latency_ms": 2100.0,
    }
    res = ingest.adapt_vllm(metrics_dict, timestamp=1780000000.0)
    assert res.model_load_events == []            # INVARIANT: never invented
    assert res.residency_observable is False
    assert res.prefix_cache_hit_rate == 0.82      # the one real signal
    obs = res.observations[0]
    assert obs.model_loaded_before_request is None  # vLLM cannot report this
    assert obs.model_load_latency_s is None
    assert obs.ttft_s == 0.31 and obs.e2e_latency_s == 2.1
    assert obs.confidence == "low"                # aggregate, not per-request


# ---------------------------------------------------------------------------
# Linkage — no fake joins
# ---------------------------------------------------------------------------

def test_linkage_container_join_on_fixtures():
    obs = ingest.import_observations(REQUESTS).records
    snaps = ingest.import_snapshots(SNAPSHOTS).records
    report = linkage.build_linkage_report(obs, snaps)
    assert report.quality == "container_join"
    assert report.attributable is True
    assert report.has_container_key is True
    # req-6 (no keys) cannot container-join → time_join only
    assert report.per_quality["time_join"] == 1
    assert report.per_quality["container_join"] == 6


def test_linkage_exact_join_via_request_id():
    obs = ingest.import_observations(REQUESTS).records
    events = ingest.import_events(EVENTS).records
    report = linkage.build_linkage_report(obs, events)
    assert report.quality == "exact_join"
    assert report.has_request_id_key is True


def test_linkage_no_join_when_no_keys_and_no_time_overlap():
    a = [RequestResidencyObservation(request_id="r", timestamp=1000.0,
                                     model_id="m", source="t")]
    b = [ModelResidencySnapshot(timestamp=9_000_000.0, source="t",
                                container_id="c", gpu_id="g")]
    report = linkage.build_linkage_report(a, b, time_tolerance_s=300.0)
    assert report.quality == "no_join"
    assert report.attributable is False


def test_linkage_refuses_mismatched_gpu_join():
    # same container_id but different gpu_id → must NOT container-join
    obs = RequestResidencyObservation(request_id="r", timestamp=100.0, model_id="m",
                                      source="t", container_id="pod-x", gpu_id="gpu-0")
    snap = ModelResidencySnapshot(timestamp=100.0, source="t",
                                  container_id="pod-x", gpu_id="gpu-9")
    rec, q = linkage.best_match(obs, [snap], time_tolerance_s=300.0)
    assert q == "no_join" and rec is None
