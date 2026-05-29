"""Part E — interactive queue / proxy / prefix candidate actions.

Proves the constraint engine:
  * emits a queue-relief candidate that advertises scale / prewarm / reserve
    capacity variants under a real queue surge;
  * reroutes (and avoids a useless scale-up) when the ingress PROXY — not
    replica count — is the bottleneck;
  * preserves prefix-cache affinity (blocks the cache-destroying energy move)
    when the hit rate is high, and allows the move when the hit rate is low.

These exercise the generators directly (no pinned benchmark numbers) plus the
full engine pipeline for the block/avoid behaviours.
"""

from __future__ import annotations

from datetime import datetime, timezone

from aurelius.constraints.classifier import ConstraintConfig
from aurelius.constraints.engine import (
    ConstraintAwareEngine,
    _gen_energy,
    _gen_queue,
)
from aurelius.sla.actions import ActionType
from aurelius.state.models import (
    ClusterState,
    ConstraintAssessment,
    ConstraintType,
    EnergyState,
    GPUState,
    InferenceServiceState,
    NodeState,
    Provenance,
    RegionState,
)

NOW = datetime(2026, 2, 1, tzinfo=timezone.utc)


def _p():
    return Provenance(source="test", fetched_at=NOW, confidence="high")


def _gpu(uuid, node, region, util=None):
    return GPUState(gpu_uuid=uuid, node_id=node, region=region, timestamp=NOW,
                    provenance=_p(), gpu_index=0, util_pct=util)


def _svc(sid, region, **kw):
    return InferenceServiceState(service_id=sid, engine="vllm", timestamp=NOW,
                                 provenance=_p(), region=region, **kw)


def _region(rid, *, price=None, pct=None, spare=None, services=None, gpus=None):
    energy = (EnergyState(region=rid, timestamp=NOW, provenance=_p(),
                          price_per_mwh=price, price_percentile=pct)
              if price is not None else None)
    node_gpus = {g.gpu_uuid: g for g in (gpus or [])}
    n = NodeState(node_id=f"{rid}-n0", region=rid, timestamp=NOW, provenance=_p(),
                  gpu_capacity=4, gpu_allocatable=4, gpu_allocated=2, gpus=node_gpus)
    return RegionState(region=rid, timestamp=NOW, provenance=_p(), nodes={n.node_id: n},
                       services=services or {}, energy=energy, spare_capacity_pct=spare)


def _cluster(regions):
    return ClusterState(timestamp=NOW, provenance=_p(), regions=regions)


def _empty_assessment():
    return ConstraintAssessment(timestamp=NOW, provenance=_p(), region=None,
                                scores={}, binding_constraint=ConstraintType.QUEUE,
                                confidence=0.9, missing_signals=[], rationale="")


# ---------------------------------------------------------------------------
# Queue surge — scale / prewarm / reserve relief variants
# ---------------------------------------------------------------------------

def test_gen_queue_emits_scale_with_prewarm_and_reserve_variants():
    svc = _svc("s", "r", requests_waiting=80, queue_time_p95_ms=1500)
    state = _cluster({"r": _region("r", spare=30, services={"s": svc})})
    cands = _gen_queue(svc, state, _empty_assessment())
    by_type = {c.action_type: c for c in cands}
    assert ActionType.SCALE_REPLICAS in by_type
    assert ActionType.SPREAD in by_type
    variants = by_type[ActionType.SCALE_REPLICAS].metadata.get("relief_variants")
    assert variants == ["scale_replicas", "prewarm_replica", "reserve_capacity_for_sla"]


# ---------------------------------------------------------------------------
# Proxy bottleneck — reroute, flag, avoid useless scale
# ---------------------------------------------------------------------------

def test_gen_queue_proxy_bottleneck_reroutes_and_avoids_scale():
    svc = _svc("s", "hot", requests_waiting=90, queue_time_p95_ms=2000,
               proxy_saturation=2.4)
    state = _cluster({
        "hot": _region("hot", spare=5, services={"s": svc},
                       gpus=[_gpu("a", "hot-n0", "hot", util=95)]),
        "cool": _region("cool", spare=70, gpus=[_gpu("b", "cool-n0", "cool", util=20)]),
    })
    cands = _gen_queue(svc, state, _empty_assessment())
    types = {c.action_type for c in cands}
    assert ActionType.SCALE_REPLICAS not in types, "must not scale when proxy is the bottleneck"
    assert ActionType.REROUTE in types
    reroute = next(c for c in cands if c.action_type == ActionType.REROUTE)
    assert reroute.target_region == "cool"
    assert reroute.metadata.get("flag") == "proxy_bottleneck"


def test_gen_queue_replica_bound_when_proxy_not_saturated():
    svc = _svc("s", "r", requests_waiting=80, queue_time_p95_ms=1500,
               proxy_saturation=0.4)  # proxy healthy
    state = _cluster({"r": _region("r", spare=30, services={"s": svc})})
    types = {c.action_type for c in _gen_queue(svc, state, _empty_assessment())}
    assert ActionType.SCALE_REPLICAS in types  # replica-bound path


def test_engine_proxy_bottleneck_does_not_recommend_scale():
    svc = _svc("s", "hot", requests_waiting=120, queue_time_p95_ms=3000,
               p99_latency_ms=4000, proxy_saturation=3.0,
               priority_tier="latency_sensitive", latency_sensitive=True)
    state = _cluster({
        "hot": _region("hot", spare=5, services={"s": svc},
                       gpus=[_gpu("a", "hot-n0", "hot", util=97)]),
        "cool": _region("cool", spare=75, gpus=[_gpu("b", "cool-n0", "cool", util=15)]),
    })
    engine = ConstraintAwareEngine(classifier_config=ConstraintConfig(hysteresis_count=1))
    recs = [r for r in engine.run(state).recommendations if r.workload_id == "s"]
    assert len(recs) == 1
    assert recs[0].action_type != ActionType.SCALE_REPLICAS.value


# ---------------------------------------------------------------------------
# Prefix affinity — preserve cache / block cache-destroying energy move
# ---------------------------------------------------------------------------

def _energy_state(hit_rate):
    svc = _svc("chat", "expensive", prefix_cache_hit_rate=hit_rate,
               p99_latency_ms=300, priority_tier="standard")
    return _cluster({
        "expensive": _region("expensive", price=200, pct=98, spare=60,
                             services={"chat": svc},
                             gpus=[_gpu("a", "expensive-n0", "expensive", util=60)]),
        "cheap": _region("cheap", price=40, pct=8, spare=70,
                        gpus=[_gpu("b", "cheap-n0", "cheap", util=30)]),
    })


def test_gen_energy_preserves_affinity_when_hit_rate_high():
    state = _energy_state(0.85)
    svc = state.regions["expensive"].services["chat"]
    cands = _gen_energy(svc, state, _empty_assessment())
    types = {c.action_type for c in cands}
    assert ActionType.CHOOSE_CHEAPER_REGION not in types, \
        "high prefix-cache affinity must block the cache-destroying region move"
    assert types == {ActionType.DEFER}
    assert cands[0].metadata.get("preserve_affinity") is True


def test_gen_energy_allows_move_when_hit_rate_low():
    state = _energy_state(0.15)
    svc = state.regions["expensive"].services["chat"]
    types = {c.action_type for c in _gen_energy(svc, state, _empty_assessment())}
    assert ActionType.CHOOSE_CHEAPER_REGION in types  # low affinity ⇒ move allowed


def test_engine_blocks_cache_destroying_energy_move():
    state = _energy_state(0.9)
    engine = ConstraintAwareEngine(classifier_config=ConstraintConfig(hysteresis_count=1))
    recs = [r for r in engine.run(state).recommendations if r.workload_id == "chat"]
    assert len(recs) == 1
    # The engine must not move a high-affinity service across regions for energy.
    assert recs[0].target_region in (None, "expensive")
    assert recs[0].action_type != ActionType.CHOOSE_CHEAPER_REGION.value
