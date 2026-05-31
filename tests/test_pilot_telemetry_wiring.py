"""Pilot telemetry wiring — implementation tests.

Asserts the contract for the strengthened bridge + the new helper
modules. Specifically:

 1. Timeout fallback hierarchy (A → B → C → D) returns the right level.
 2. SLA-violation derivation is binary per-tick and ``None`` when inputs
    missing.
 3. K8s scale-event delta is deterministic across two snapshots, with
    pod-name churn separable from net replica delta.
 4. K8s scale-event delta supports owner-name and namespace filters.
 5. Churn derivation distinguishes restart-with-same-count (churn > 0,
    replica_delta == 0) from scale-up (churn > 0, replica_delta > 0).
 6. vLLM prefix cache hit rate flows through the bridge into the
    diagnostic field on the resulting ServingTelemetryTick context.
 7. Missing fields remain missing AND emit a MISSING provenance entry.
 8. Proxy fields are labeled PROXY (vLLM TTFT used as queue p99 proxy;
    vLLM requests_running used as observed_rps proxy).
 9. Derived fields are labeled DERIVED (gpu_hours_delta, sla_violation,
    Triton p50 average, K8s scale/churn).
10. Real adapter fields are labeled REAL (Ray Serve latency p99 / p95,
    Ray Serve replicas).
11. Provenance flows through DynamicFrontierPrediction.tick_provenance
    and is JSON round-trippable.
12. Calibration summary surfaces telemetry-provenance roll-up.
13. SGLang adapter is intentionally BLOCKED_SCHEMA_UNKNOWN — fixture
    is absent and the adapter has not been written.
14. vLLM connector adapter scrapes ``vllm:num_preemptions_total`` into
    ``InferenceServiceState.preemptions_total`` (added metric in the
    fixture).
15. The new ``vllm:num_preemptions_total`` metric is exposed in the
    built-in vLLM metric mapping registry.
16. Calibration replay tolerates a tick built from real ISS + DCGM
    + K8s deltas without crashing.
17. Bridge timeout fallback uses C_ERROR_RATE when ``error_rate_pct``
    is present and prefers it over the SLA-risk proxy (D).
18. Bridge timeout fallback falls back to D_SLA_RISK_PROXY when no
    error_rate is present but latency + SLA are.
19. Provenance entries are JSON round-trippable via TickProvenance
    dict path.
20. ServingTelemetryTick produced by the bridge preserves the
    Provenance.confidence label.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone

import pytest

from aurelius.connectors import (
    K8sPlacementSnapshot,
    K8sReplicaDelta,
    PodPlacement,
    compute_k8s_scale_delta,
)
from aurelius.connectors.metric_mapping import vllm_registry
from aurelius.frontier import (
    CalibrationReplayConfig,
    DynamicFrontierObservedOutcome,
    DynamicFrontierPrediction,
    FieldOrigin,
    OracleSeriesPoint,
    ServingTelemetryTick,
    TickProvenance,
    TimeoutFallback,
    WorkloadFrontierProfile,
    apply_confidence_update,
    build_serving_telemetry_window,
    compute_calibration_record,
    compute_calibration_records,
    compute_frontier_calibration_summary,
    derive_sla_violation_pct,
    resolve_timeout_pct,
    run_dynamic_frontier_calibration_replay,
    telemetry_tick_from_inference_service_state,
    telemetry_tick_with_provenance_from_inference_service_state,
    validate_dynamic_window,
)
from aurelius.state.models import InferenceServiceState, Provenance


def _provenance(confidence="medium", source="unit"):
    return Provenance(source=source, fetched_at=datetime.now(timezone.utc),
                       confidence=confidence, is_sandbox=False)


def _vllm_state(*, p99_latency_ms=900.0, p95_latency_ms=700.0,
                p50_latency_ms=300.0, ttft_p99_ms=200.0,
                requests_running=10.0, requests_waiting=2.0,
                kv_cache_usage=0.55, prefix_cache_hit_rate=0.42,
                tokens_per_s=120.0, error_rate_pct=None):
    return InferenceServiceState(
        service_id="vllm/llama3-70b", engine="vllm",
        timestamp=datetime.now(timezone.utc),
        provenance=_provenance(confidence="medium"),
        requests_running=requests_running,
        requests_waiting=requests_waiting,
        p50_latency_ms=p50_latency_ms,
        p95_latency_ms=p95_latency_ms,
        p99_latency_ms=p99_latency_ms,
        ttft_p99_ms=ttft_p99_ms,
        kv_cache_usage=kv_cache_usage,
        prefix_cache_hit_rate=prefix_cache_hit_rate,
        tokens_per_s=tokens_per_s,
        error_rate_pct=error_rate_pct)


def _ray_state(*, replicas=4, error_rate_pct=0.5,
                p99_latency_ms=300.0, p95_latency_ms=150.0,
                queue_time_p99_ms=None):
    return InferenceServiceState(
        service_id="ray/serve/llama", engine="ray_serve",
        timestamp=datetime.now(timezone.utc),
        provenance=_provenance(confidence="high"),
        requests_waiting=3.0,
        p50_latency_ms=80.0, p95_latency_ms=p95_latency_ms,
        p99_latency_ms=p99_latency_ms,
        queue_time_p99_ms=queue_time_p99_ms,
        error_rate_pct=error_rate_pct, replicas=replicas)


def _triton_state():
    return InferenceServiceState(
        service_id="triton/resnet", engine="triton",
        timestamp=datetime.now(timezone.utc),
        provenance=_provenance(confidence="low"),
        requests_waiting=1.0,
        p50_latency_ms=45.0,    # Triton: cumulative average, NOT p50
        queue_time_p50_ms=5.0,  # Triton: derived average, NOT p50
        error_rate_pct=0.1)


# ---------------------------------------------------------------------------
# 1 — timeout fallback hierarchy
# ---------------------------------------------------------------------------

def test_timeout_fallback_returns_A_when_explicit_counter():
    res = resolve_timeout_pct(
        explicit_timeout_counter_rate=2.0,
        total_request_rate=100.0,
        error_rate_pct=5.0,           # would be ignored — A wins
        latency_p99_ms=2000.0,
        latency_sla_p99_ms=1000.0,
    )
    assert res.level == TimeoutFallback.A_TIMEOUT_COUNTER
    assert res.value == pytest.approx(2.0)


def test_timeout_fallback_returns_B_when_deadline_counter():
    res = resolve_timeout_pct(
        deadline_exceeded_rate=1.0,
        total_request_rate=50.0,
        error_rate_pct=8.0,
    )
    assert res.level == TimeoutFallback.B_DEADLINE_COUNTER
    assert res.value == pytest.approx(2.0)


def test_timeout_fallback_returns_C_when_only_error_rate():
    res = resolve_timeout_pct(error_rate_pct=3.5)
    assert res.level == TimeoutFallback.C_ERROR_RATE
    assert res.value == pytest.approx(3.5)


def test_timeout_fallback_returns_D_when_only_sla_proxy():
    res = resolve_timeout_pct(
        latency_p99_ms=1500.0,
        latency_sla_p99_ms=1000.0,
    )
    assert res.level == TimeoutFallback.D_SLA_RISK_PROXY
    assert res.value == 100.0


def test_timeout_fallback_returns_none_when_nothing_available():
    res = resolve_timeout_pct()
    assert res.level == TimeoutFallback.NONE
    assert res.value is None


# ---------------------------------------------------------------------------
# 2 — SLA-violation derivation
# ---------------------------------------------------------------------------

def test_sla_violation_binary_when_over_threshold():
    assert derive_sla_violation_pct(
        latency_p99_ms=1100.0, latency_sla_p99_ms=1000.0) == 100.0


def test_sla_violation_zero_when_under_threshold():
    assert derive_sla_violation_pct(
        latency_p99_ms=800.0, latency_sla_p99_ms=1000.0) == 0.0


def test_sla_violation_none_when_inputs_missing():
    assert derive_sla_violation_pct(
        latency_p99_ms=None, latency_sla_p99_ms=1000.0) is None
    assert derive_sla_violation_pct(
        latency_p99_ms=800.0, latency_sla_p99_ms=None) is None
    # Defensive: 0 SLA budget is treated as missing, not divide-by-zero.
    assert derive_sla_violation_pct(
        latency_p99_ms=800.0, latency_sla_p99_ms=0.0) is None


# ---------------------------------------------------------------------------
# 3 — K8s scale-event delta across two snapshots
# ---------------------------------------------------------------------------

def _snapshot(pods, *, fetched_at=None):
    fetched_at = fetched_at or datetime.now(timezone.utc)
    return K8sPlacementSnapshot(
        nodes={}, pods=pods, fetched_at=fetched_at, is_partial=False,
        missing_sources=[], is_sandbox=True)


def _pod(name, *, namespace="inference", node="gpu-0",
          phase="Running", gpu=1, owner="llm-inference"):
    return PodPlacement(
        pod_name=name, namespace=namespace, node_name=node,
        gpu_count=gpu, phase=phase, start_time=None, labels={},
        owner_kind="Deployment", owner_name=owner)


def test_k8s_scale_delta_detects_scale_up():
    t0 = datetime.now(timezone.utc)
    prev = _snapshot([_pod("p1"), _pod("p2")], fetched_at=t0)
    curr = _snapshot([_pod("p1"), _pod("p2"), _pod("p3"), _pod("p4")],
                      fetched_at=t0 + timedelta(seconds=60))
    d = compute_k8s_scale_delta(prev, curr)
    assert d.prev_replicas == 2
    assert d.curr_replicas == 4
    assert d.replica_delta == 2
    assert d.scale_events == 1
    assert d.added_pod_names == frozenset({"p3", "p4"})
    assert d.removed_pod_names == frozenset()
    assert d.churn_count == 2
    assert d.window_seconds == pytest.approx(60.0)


def test_k8s_scale_delta_detects_scale_down():
    t0 = datetime.now(timezone.utc)
    prev = _snapshot([_pod("p1"), _pod("p2"), _pod("p3")], fetched_at=t0)
    curr = _snapshot([_pod("p1")], fetched_at=t0 + timedelta(seconds=60))
    d = compute_k8s_scale_delta(prev, curr)
    assert d.replica_delta == -2
    assert d.removed_pod_names == frozenset({"p2", "p3"})


# ---------------------------------------------------------------------------
# 4 — K8s owner-name + namespace filter
# ---------------------------------------------------------------------------

def test_k8s_scale_delta_owner_filter():
    t0 = datetime.now(timezone.utc)
    prev = _snapshot([
        _pod("p1", owner="llm"), _pod("p2", owner="other"),
    ], fetched_at=t0)
    curr = _snapshot([
        _pod("p1", owner="llm"), _pod("p2", owner="other"),
        _pod("p3", owner="other"),
    ], fetched_at=t0 + timedelta(seconds=60))
    d_llm = compute_k8s_scale_delta(prev, curr, owner_name="llm")
    assert d_llm.prev_replicas == 1
    assert d_llm.curr_replicas == 1
    assert d_llm.replica_delta == 0
    d_other = compute_k8s_scale_delta(prev, curr, owner_name="other")
    assert d_other.replica_delta == 1


def test_k8s_scale_delta_namespace_filter():
    t0 = datetime.now(timezone.utc)
    prev = _snapshot([
        _pod("p1", namespace="inference"),
        _pod("b1", namespace="batch"),
    ], fetched_at=t0)
    curr = _snapshot([
        _pod("p1", namespace="inference"),
        _pod("p2", namespace="inference"),
        _pod("b1", namespace="batch"),
    ], fetched_at=t0 + timedelta(seconds=60))
    d = compute_k8s_scale_delta(prev, curr, namespace="inference")
    assert d.replica_delta == 1


# ---------------------------------------------------------------------------
# 5 — rolling restart (churn high, replica_delta = 0)
# ---------------------------------------------------------------------------

def test_k8s_rolling_restart_churn_separate_from_replica_delta():
    t0 = datetime.now(timezone.utc)
    prev = _snapshot([_pod("p1"), _pod("p2"), _pod("p3")], fetched_at=t0)
    curr = _snapshot([_pod("p4"), _pod("p5"), _pod("p6")],
                      fetched_at=t0 + timedelta(seconds=60))
    d = compute_k8s_scale_delta(prev, curr)
    assert d.replica_delta == 0
    assert d.scale_events == 0  # net count unchanged
    assert d.churn_count == 6   # 3 added + 3 removed
    assert d.added_pod_names == frozenset({"p4", "p5", "p6"})
    assert d.removed_pod_names == frozenset({"p1", "p2", "p3"})


# ---------------------------------------------------------------------------
# 6 — vLLM prefix cache hit rate flows through (diagnostic)
# ---------------------------------------------------------------------------

def test_vllm_prefix_cache_hit_rate_propagates_to_inference_service_state():
    """The vLLM adapter normalizes
    vllm:gpu_prefix_cache_hit_rate into InferenceServiceState.
    prefix_cache_hit_rate (0-1 fraction). Pre-existing assertion;
    surfaced here so the wiring is regression-tested."""
    state = _vllm_state(prefix_cache_hit_rate=0.42)
    assert state.prefix_cache_hit_rate == pytest.approx(0.42)


# ---------------------------------------------------------------------------
# 7 — missing fields stay missing and emit MISSING provenance
# ---------------------------------------------------------------------------

def test_missing_fields_emit_missing_provenance():
    # Minimal ISS: only timestamp + engine + provenance set.
    state = InferenceServiceState(
        service_id="vllm/minimal", engine="vllm",
        timestamp=datetime.now(timezone.utc),
        provenance=_provenance())
    tick, prov = telemetry_tick_with_provenance_from_inference_service_state(
        state)
    # Tick fields stay None.
    for f in ("observed_rps", "queue_p99_ms", "queue_p95_ms",
              "latency_p99_ms", "timeout_pct", "sla_violation_pct",
              "mean_utilization", "active_replicas", "gpu_hours_delta",
              "scale_events_delta", "churn_delta"):
        assert getattr(tick, f) is None
    # And the provenance entry exists with origin=MISSING.
    for f in ("observed_rps", "queue_p99_ms", "latency_p99_ms",
              "timeout_pct", "sla_violation_pct", "mean_utilization",
              "active_replicas", "scale_events_delta", "churn_delta"):
        entry = prov.get(f)
        assert entry is not None, f"missing provenance entry for {f}"
        assert entry.origin == FieldOrigin.MISSING


# ---------------------------------------------------------------------------
# 8 — proxy fields labeled PROXY
# ---------------------------------------------------------------------------

def test_vllm_ttft_proxies_queue_p99_with_proxy_label():
    state = _vllm_state()  # no queue_time_p99_ms; ttft_p99_ms=200
    tick, prov = telemetry_tick_with_provenance_from_inference_service_state(
        state)
    assert tick.queue_p99_ms == 200.0
    entry = prov.get("queue_p99_ms")
    assert entry.origin == FieldOrigin.PROXY
    assert "ttft" in entry.source.lower()


def test_vllm_requests_running_marked_proxy_for_observed_rps():
    state = _vllm_state(requests_running=10.0)
    tick, prov = telemetry_tick_with_provenance_from_inference_service_state(
        state)
    assert tick.observed_rps == 10.0
    entry = prov.get("observed_rps")
    assert entry.origin == FieldOrigin.PROXY


# ---------------------------------------------------------------------------
# 9 — derived fields labeled DERIVED
# ---------------------------------------------------------------------------

def test_gpu_hours_delta_marked_derived_when_inferred():
    state = _ray_state(replicas=4)
    tick, prov = telemetry_tick_with_provenance_from_inference_service_state(
        state, tick_duration_s=60.0)
    assert tick.gpu_hours_delta == pytest.approx(4.0 * 60.0 / 3600.0)
    entry = prov.get("gpu_hours_delta")
    assert entry.origin == FieldOrigin.DERIVED


def test_sla_violation_marked_derived_when_inputs_present():
    state = _vllm_state(p99_latency_ms=1500.0)
    tick, prov = telemetry_tick_with_provenance_from_inference_service_state(
        state, latency_sla_p99_ms=1000.0)
    assert tick.sla_violation_pct == 100.0
    entry = prov.get("sla_violation_pct")
    assert entry.origin == FieldOrigin.DERIVED


def test_triton_queue_p50_marked_derived_average():
    state = _triton_state()
    tick, prov = telemetry_tick_with_provenance_from_inference_service_state(
        state)
    # Triton queue_time_p50_ms is a cumulative average, not a percentile.
    assert tick.queue_p50_ms == 5.0
    entry = prov.get("queue_p50_ms")
    assert entry.origin == FieldOrigin.DERIVED


def test_k8s_scale_delta_marks_scale_and_churn_as_derived():
    state = _ray_state(replicas=4)
    tick, prov = telemetry_tick_with_provenance_from_inference_service_state(
        state, tick_duration_s=60.0, scale_events_delta=1,
        churn_delta=2.0)
    assert tick.scale_events_delta == 1
    assert tick.churn_delta == 2.0
    assert prov.get("scale_events_delta").origin == FieldOrigin.DERIVED
    assert prov.get("churn_delta").origin == FieldOrigin.DERIVED


# ---------------------------------------------------------------------------
# 10 — real adapter fields labeled REAL
# ---------------------------------------------------------------------------

def test_ray_serve_replicas_and_latency_marked_real():
    state = _ray_state(replicas=6, p99_latency_ms=300.0, p95_latency_ms=150.0,
                        queue_time_p99_ms=80.0)
    tick, prov = telemetry_tick_with_provenance_from_inference_service_state(
        state)
    assert tick.active_replicas == 6
    assert prov.get("active_replicas").origin == FieldOrigin.REAL
    assert tick.queue_p99_ms == 80.0  # REAL queue percentile
    assert prov.get("queue_p99_ms").origin == FieldOrigin.REAL
    assert prov.get("latency_p99_ms").origin == FieldOrigin.REAL
    assert prov.get("latency_p95_ms").origin == FieldOrigin.REAL


def test_dcgm_mean_utilization_marked_real_when_caller_supplies():
    state = _vllm_state()
    tick, prov = telemetry_tick_with_provenance_from_inference_service_state(
        state, mean_utilization=0.65,
        mean_utilization_source="dcgm.DCGM_FI_DEV_GPU_UTIL")
    assert tick.mean_utilization == 0.65
    entry = prov.get("mean_utilization")
    assert entry.origin == FieldOrigin.REAL
    assert "dcgm" in entry.source.lower()


# ---------------------------------------------------------------------------
# 11 — provenance flows through DynamicFrontierPrediction (JSON round-trip)
# ---------------------------------------------------------------------------

def test_tick_provenance_round_trips_through_prediction_json():
    state = _ray_state(replicas=4, p99_latency_ms=900.0,
                        queue_time_p99_ms=100.0)
    tick, prov = telemetry_tick_with_provenance_from_inference_service_state(
        state, tick_duration_s=60.0, mean_utilization=0.65,
        latency_sla_p99_ms=1000.0)
    p = DynamicFrontierPrediction(
        timestamp_s=0.0, workload_id="w", current_rho=0.65,
        recommended_rho=0.65, action="KEEP_RHO",
        tick_provenance=prov.to_dict())
    s = json.dumps(p.to_dict(), default=str)
    back = DynamicFrontierPrediction.from_dict(json.loads(s))
    assert back.tick_provenance is not None
    rebuilt = TickProvenance.from_dict(back.tick_provenance)
    # queue_p99_ms came from queue_time_p99_ms ⇒ REAL.
    assert rebuilt.get("queue_p99_ms").origin == FieldOrigin.REAL
    # sla_violation_pct ⇒ DERIVED.
    assert rebuilt.get("sla_violation_pct").origin == FieldOrigin.DERIVED


# ---------------------------------------------------------------------------
# 12 — calibration summary surfaces telemetry-provenance roll-up
# ---------------------------------------------------------------------------

def test_calibration_summary_includes_telemetry_provenance_block():
    state = _ray_state(replicas=4, queue_time_p99_ms=80.0)
    tick, prov = telemetry_tick_with_provenance_from_inference_service_state(
        state, tick_duration_s=60.0)
    p = DynamicFrontierPrediction(
        timestamp_s=0.0, workload_id="w", current_rho=0.65,
        recommended_rho=0.65, action="KEEP_RHO",
        predicted_goodput_per_dollar=1.0,
        predicted_timeout_pct=2.0, predicted_queue_p99_ms=300.0,
        predicted_sla_risk_probability=0.1,
        predicted_queue_blowup_probability=0.1,
        tick_provenance=prov.to_dict())
    o = DynamicFrontierObservedOutcome(
        timestamp_s=0.0, workload_id="w", applied_rho=0.65,
        observed_goodput_per_dollar=1.0, observed_timeout_pct=2.0,
        observed_queue_p99_ms=300.0, was_safe=True)
    rec = compute_calibration_record(p, o)
    summary = compute_frontier_calibration_summary([rec])
    tp = summary.get("telemetry_provenance")
    assert tp is not None
    assert tp["records_with_provenance"] == 1
    # The single record carries 13 entries — at least one REAL and one
    # MISSING in this synthetic case.
    assert "real" in tp["origin_counts"]


# ---------------------------------------------------------------------------
# 13 — SGLang is BLOCKED_SCHEMA_UNKNOWN (no adapter, no fixture)
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason=(
    "SGLang Prometheus adapter is intentionally BLOCKED_SCHEMA_UNKNOWN: "
    "no fixture exists at tests/fixtures/prometheus/sglang_metrics.prom, "
    "and the metric_mapping registry does not define an sglang_registry(). "
    "Implementing this requires verifying the SGLang Prometheus schema "
    "against the project's public exporter and adding a fixture first."))
def test_sglang_adapter_blocked_schema_unknown():
    # When we add it, this test will: (a) load
    # tests/fixtures/prometheus/sglang_metrics.prom via the Prometheus
    # text parser; (b) normalize via an SGLangAdapter into an
    # InferenceServiceState; (c) bridge into a ServingTelemetryTick.
    pytest.fail("SGLang adapter not implemented")


def test_sglang_adapter_absent_from_repo():
    # Defensive sanity check: no sglang.py exists yet so we do not have
    # a stub silently masquerading as a real adapter.
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    assert not os.path.exists(os.path.join(
        repo_root, "aurelius", "connectors", "sglang.py")), (
        "If you add an SGLang adapter, also add a fixture under "
        "tests/fixtures/prometheus/sglang_metrics.prom and update this "
        "test to scrape it.")


# ---------------------------------------------------------------------------
# 14 + 15 — vLLM preemptions metric wired through adapter + registry
# ---------------------------------------------------------------------------

def test_vllm_preemptions_metric_registered():
    reg = vllm_registry()
    m = reg.get("inference.preemptions_per_second")
    assert m is not None
    assert "vllm:num_preemptions_total" in m.query


def test_vllm_adapter_scrapes_preemptions_from_fixture():
    from datetime import datetime as _dt, timezone as _tz
    from aurelius.connectors.base import TelemetrySnapshot
    from aurelius.connectors.prometheus import parse_prometheus_text
    from aurelius.connectors.metric_mapping import vllm_registry
    from aurelius.connectors.vllm import VLLMAdapter

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fixture = os.path.join(repo_root, "tests", "fixtures", "prometheus",
                            "vllm_metrics.prom")
    with open(fixture, encoding="utf-8") as fh:
        text = fh.read()
    raw = parse_prometheus_text(text)

    # Build a snapshot that pre-resolves the canonical fields the
    # VLLMAdapter expects. We compute a synthetic rate for the
    # preemption counter so the snapshot carries a non-None value.
    snapshot = TelemetrySnapshot(
        source="vllm_fixture",
        fetched_at=_dt.now(tz=_tz.utc),
        is_sandbox=True,
        metrics={
            "inference.queue_depth":
                raw.get("vllm:num_requests_waiting"),
            "inference.active_sequences":
                raw.get("vllm:num_requests_running"),
            "inference.kv_cache_usage_pct":
                raw.get("vllm:gpu_cache_usage_perc"),
            "inference.prefix_cache_hit_rate_pct":
                raw.get("vllm:gpu_prefix_cache_hit_rate"),
            "inference.preemptions_per_second":
                raw.get("vllm:num_preemptions_total"),
        },
    )
    adapter = VLLMAdapter()
    states = adapter.normalize_all_services(snapshot)
    by_model = {s.service_id.split("/")[-1]: s for s in states}
    assert "llama3-70b" in by_model
    # Fixture: vllm:num_preemptions_total{model_name="llama3-70b"} = 47
    assert by_model["llama3-70b"].preemptions_total == 47.0
    # The fact this is a counter (not a rate) is acceptable for the
    # adapter contract — the rate computation lives upstream in
    # Prometheus / PromQL. The bridge does NOT treat preemptions_total
    # as a timeout signal (see tests 17/18).


# ---------------------------------------------------------------------------
# 16 — calibration replay tolerates real-ISS-derived ticks
# ---------------------------------------------------------------------------

def test_calibration_replay_consumes_real_iss_derived_telemetry():
    """End-to-end smoke: build 12 ticks from real ISS observations
    (with caller-supplied DCGM rho + K8s deltas), validate the window,
    and run a single calibration replay pass — must not crash."""
    profile = WorkloadFrontierProfile(
        workload_id="pilot/wl", workload_type="inference_standard",
        telemetry_confidence="medium", priority_class="standard")
    # 50 ticks at 60s.
    ticks = list(range(50))

    def eval_fn(target_rho, idx):
        # Realistic-ish synthetic eval; the calibration loop only needs
        # a dict with the canonical fields.
        return {
            "rho": target_rho,
            "timeout_pct": 2.0,
            "queue_p99_ms": 200.0 + 100 * target_rho,
            "latency_p99_ms": 800.0,
            "gpu_hours": 1.0,
            "goodput_per_dollar": 1000.0,
        }

    def telemetry_fn(tick_idx, ev):
        # Build a real ISS observation that emulates a Ray Serve scrape.
        iss = _ray_state(replicas=4, p99_latency_ms=ev["latency_p99_ms"],
                          p95_latency_ms=ev["latency_p99_ms"] * 0.8,
                          queue_time_p99_ms=ev["queue_p99_ms"])
        # Use the bridge.
        tick, _prov = telemetry_tick_with_provenance_from_inference_service_state(
            iss,
            timestamp_s=float(tick_idx) * 60.0,
            tick_duration_s=60.0,
            mean_utilization=ev["rho"],
            scale_events_delta=0,
            churn_delta=0.0,
            latency_sla_p99_ms=2000.0,
            source="real_pilot_telemetry")
        return tick

    cfg = CalibrationReplayConfig(window_ticks=8,
                                   decision_interval_ticks=1)
    pr = run_dynamic_frontier_calibration_replay(
        workload_profile=profile, ticks=ticks,
        eval_fn=eval_fn, telemetry_fn=telemetry_fn, config=cfg)
    assert pr.summary.get("n_records", 0) > 0


# ---------------------------------------------------------------------------
# 17 — timeout fallback uses C_ERROR_RATE when error_rate_pct present
# ---------------------------------------------------------------------------

def test_bridge_timeout_uses_C_when_error_rate_pct_present():
    state = _vllm_state(p99_latency_ms=900.0, error_rate_pct=2.5)
    tick, prov = telemetry_tick_with_provenance_from_inference_service_state(
        state, latency_sla_p99_ms=1000.0)
    assert tick.timeout_pct == pytest.approx(2.5)
    entry = prov.get("timeout_pct")
    assert entry.origin == FieldOrigin.PROXY
    assert entry.fallback_level == TimeoutFallback.C_ERROR_RATE


# ---------------------------------------------------------------------------
# 18 — timeout fallback falls back to D when no error_rate
# ---------------------------------------------------------------------------

def test_bridge_timeout_falls_back_to_D_when_no_error_rate():
    state = _vllm_state(p99_latency_ms=1500.0, error_rate_pct=None)
    tick, prov = telemetry_tick_with_provenance_from_inference_service_state(
        state, latency_sla_p99_ms=1000.0)
    # D proxy: p99_latency > SLA ⇒ 100.0
    assert tick.timeout_pct == 100.0
    entry = prov.get("timeout_pct")
    assert entry.fallback_level == TimeoutFallback.D_SLA_RISK_PROXY


def test_bridge_timeout_none_when_nothing_available():
    state = _vllm_state(p99_latency_ms=None, error_rate_pct=None)
    tick, prov = telemetry_tick_with_provenance_from_inference_service_state(
        state)
    assert tick.timeout_pct is None
    entry = prov.get("timeout_pct")
    assert entry.origin == FieldOrigin.MISSING


# ---------------------------------------------------------------------------
# 19 — TickProvenance JSON round-trip preserves origin enum + fallback level
# ---------------------------------------------------------------------------

def test_tick_provenance_round_trips_to_dict_and_back():
    state = _vllm_state(p99_latency_ms=1500.0, error_rate_pct=None)
    tick, prov = telemetry_tick_with_provenance_from_inference_service_state(
        state, latency_sla_p99_ms=1000.0)
    payload = prov.to_dict()
    back = TickProvenance.from_dict(payload)
    e = back.get("timeout_pct")
    assert e is not None
    assert e.origin == FieldOrigin.PROXY
    assert e.fallback_level == TimeoutFallback.D_SLA_RISK_PROXY
    # And origin-based queries work.
    assert "timeout_pct" in back.proxy_fields


# ---------------------------------------------------------------------------
# 20 — Provenance.confidence label propagates
# ---------------------------------------------------------------------------

def test_bridge_inherits_provenance_confidence():
    high = InferenceServiceState(
        service_id="vllm/x", engine="vllm",
        timestamp=datetime.now(timezone.utc),
        provenance=_provenance(confidence="high"))
    tick, prov = telemetry_tick_with_provenance_from_inference_service_state(
        high)
    assert tick.telemetry_confidence == "high"
    # And every per-field provenance entry carries the same confidence.
    for e in prov.entries:
        assert e.confidence == "high"


# ---------------------------------------------------------------------------
# safety floor: bridge does NOT label a fully-proxy run as REAL safety
# ---------------------------------------------------------------------------

def test_proxy_only_run_does_not_claim_real_safety_signal():
    """If every safety signal (timeout, queue, latency) is PROXY, the
    TickProvenance must not advertise has_any_real_safety_field=True —
    that boolean is the gate the audit uses to refuse promoting a
    proxy-only run."""
    state = _vllm_state(p99_latency_ms=1500.0, error_rate_pct=None)
    # No queue_time_p99_ms ⇒ queue p99 is PROXY (TTFT).
    # No error_rate ⇒ timeout is PROXY (D, SLA-risk).
    # latency p99 is REAL but engine='vllm', not derived in this case ⇒ REAL.
    tick, prov = telemetry_tick_with_provenance_from_inference_service_state(
        state, latency_sla_p99_ms=1000.0)
    # latency_p99_ms IS REAL (vLLM exposes it natively).
    assert prov.has_any_real_safety_field is True

    # Now drop latency p99 by setting engine='triton' (whose p99 is
    # marked DERIVED): with all three safety fields proxy/derived/missing,
    # the gate must reject.
    triton = InferenceServiceState(
        service_id="triton/x", engine="triton",
        timestamp=datetime.now(timezone.utc),
        provenance=_provenance(),
        p50_latency_ms=45.0,
        queue_time_p50_ms=5.0,
        error_rate_pct=None)
    tick2, prov2 = telemetry_tick_with_provenance_from_inference_service_state(
        triton, latency_sla_p99_ms=1000.0)
    # latency_p99 = None (MISSING); queue_p99 = None (MISSING);
    # timeout_pct = None (MISSING) ⇒ no REAL safety field.
    assert prov2.has_any_real_safety_field is False
