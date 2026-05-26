"""Multi-constraint engine behavior tests (Mission 2).

Recreates the audit's adversarial harness (cases A-G) as committed tests and
adds tests proving the engine consumes the FULL constraint score vector — not
just the single binding label — for both candidate generation and safety.

Key invariants:
- When an SLA-risk constraint (latency/queue/thermal/memory) is materially
  active, the engine protects it and does not chase energy/cost migrations.
- An action that would worsen a materially-active SLA-risk constraint is
  rejected even when that constraint is not the binding one.
- A migration to a critically-low / unverifiable destination is blocked
  regardless of how large the gross energy savings are.
"""

from __future__ import annotations

from datetime import datetime, timezone

from aurelius.constraints.classifier import ConstraintConfig
from aurelius.constraints.engine import ConstraintAwareEngine
from aurelius.sla.actions import ActionType
from aurelius.state.models import (
    ClusterState,
    EnergyState,
    GPUState,
    InferenceServiceState,
    NodeState,
    Provenance,
    RegionState,
    ThermalState,
    TopologyState,
)

NOW = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)
_MIGRATION = {
    ActionType.CHOOSE_CHEAPER_REGION.value,
    ActionType.MIGRATE.value,
    ActionType.CHOOSE_LOWER_CARBON_REGION.value,
}


def _p():
    return Provenance(source="test", fetched_at=NOW, confidence="high")


def _gpu(uuid, node, region, temp=None, util=None):
    return GPUState(gpu_uuid=uuid, node_id=node, region=region, timestamp=NOW,
                    provenance=_p(), gpu_index=0, temp_c=temp, util_pct=util)


def _svc(sid, region, p99=None, ttft=None, qdepth=None, qwait=None):
    return InferenceServiceState(service_id=sid, engine="vllm", timestamp=NOW,
                                 provenance=_p(), region=region, p99_latency_ms=p99,
                                 ttft_p99_ms=ttft, requests_waiting=qdepth,
                                 queue_time_p95_ms=qwait)


def _region(rid, *, price=None, pct=None, temp=None, throttling=False, spare=None,
            services=None, gpus=None, topo=None, with_node=True):
    energy = EnergyState(region=rid, timestamp=NOW, provenance=_p(), price_per_mwh=price,
                         price_percentile=pct) if price is not None else None
    thermal = ThermalState(region=rid, timestamp=NOW, provenance=_p(), max_gpu_temp_c=temp,
                           throttling_gpu_count=1 if throttling else 0,
                           total_gpu_count=4) if (temp is not None or throttling) else None
    topology = None
    if topo is not None:
        topology = TopologyState(node_id=f"{rid}-n0", timestamp=NOW, provenance=_p(),
                                 gpu_uuids=tuple(g.gpu_uuid for g in (gpus or [])),
                                 pair_levels={}, numa_affinity={}, interconnect_class=topo)
    nodes = {}
    if with_node:
        node_gpus = {g.gpu_uuid: g for g in (gpus or [])}
        n = NodeState(node_id=f"{rid}-n0", region=rid, timestamp=NOW, provenance=_p(),
                      gpu_capacity=4, gpu_allocatable=4, gpu_allocated=2, gpus=node_gpus)
        nodes[n.node_id] = n
    return RegionState(region=rid, timestamp=NOW, provenance=_p(), nodes=nodes,
                       services=services or {}, energy=energy, thermal=thermal,
                       topology=topology, spare_capacity_pct=spare)


def _cluster(regions, *, is_partial=False, missing=None):
    return ClusterState(timestamp=NOW, provenance=_p(), regions=regions,
                        is_partial=is_partial, missing_sources=missing or [])


def _engine():
    return ConstraintAwareEngine(classifier_config=ConstraintConfig(hysteresis_count=1))


def _run(state):
    return _engine().run(state)


def _rec_for(result, svc_id):
    recs = [r for r in result.recommendations if r.workload_id == svc_id]
    assert len(recs) == 1, f"expected 1 rec for {svc_id}, got {len(recs)}"
    return recs[0]


# ---------------------------------------------------------------------------
# Adversarial cases A-G
# ---------------------------------------------------------------------------

class TestAdversarialMultiConstraint:
    def test_case_a_energy_cheap_but_latency_near_sla_suppresses_migration(self):
        svc = _svc("svcA", "expensive", ttft=1900, p99=1900)
        state = _cluster({
            "expensive": _region("expensive", price=200, pct=98, spare=60,
                                  services={"svcA": svc}, gpus=[_gpu("a", "expensive-n0", "expensive", util=80)]),
            "cheap": _region("cheap", price=40, pct=10, spare=70,
                             gpus=[_gpu("b", "cheap-n0", "cheap", util=40)]),
        })
        rec = _rec_for(_run(state), "svcA")
        assert rec.action_type not in _MIGRATION, "must not migrate a latency-critical workload for energy"
        assert rec.target_region is None

    def test_case_b_energy_plus_queue_surge_prioritizes_queue(self):
        svc = _svc("svcB", "hot", qdepth=90, qwait=1500)
        state = _cluster({
            "hot": _region("hot", price=220, pct=99, spare=5, services={"svcB": svc},
                           gpus=[_gpu("c", "hot-n0", "hot", util=95)]),
            "cool": _region("cool", price=45, pct=12, spare=70,
                            gpus=[_gpu("d", "cool-n0", "cool", util=30)]),
        })
        rec = _rec_for(_run(state), "svcB")
        # Do not chase energy: no cross-region migration; queue relief or keep.
        assert rec.action_type not in _MIGRATION
        assert rec.action_type in {ActionType.SCALE_REPLICAS.value, ActionType.SPREAD.value,
                                   ActionType.KEEP.value}

    def test_case_c_utilization_with_secondary_thermal_never_consolidates(self):
        # util-dominant binding, but a real secondary thermal signal (80C ≈ 0.26).
        gpus = [_gpu("e", "r-n0", "r", temp=80, util=12), _gpu("f", "r-n0", "r", temp=80, util=12)]
        svc = _svc("svcC", "r")
        state = _cluster({"r": _region("r", temp=80, spare=85, services={"svcC": svc}, gpus=gpus)})
        result = _run(state)
        for rec in result.recommendations:
            assert rec.action_type != ActionType.CONSOLIDATE.value, \
                "must not consolidate into warm nodes when thermal is a live secondary constraint"

    def test_case_d_topology_with_active_thermal_protects_thermal(self):
        gpus = [_gpu("g", "r-n0", "r", temp=90, util=60)]
        svc = _svc("svcD", "r")
        state = _cluster({"r": _region("r", temp=90, throttling=True, topo="pcie",
                                       services={"svcD": svc}, gpus=gpus)})
        rec = _rec_for(_run(state), "svcD")
        # Must not pursue a topology placement change that worsens active thermal.
        assert rec.action_type not in {ActionType.CHANGE_PLACEMENT.value, ActionType.CONSOLIDATE.value}

    def test_case_e_migration_to_full_destination_blocked(self):
        svc = _svc("svcE", "src", p99=300)  # no SLA-risk active; high source spare
        state = _cluster({
            "src": _region("src", price=200, pct=97, spare=80, services={"svcE": svc},
                           gpus=[_gpu("h", "src-n0", "src", util=70)]),
            "dst": _region("dst", price=40, pct=8, spare=3,  # critically low
                           gpus=[_gpu("i", "dst-n0", "dst", util=98)]),
        })
        result = _run(state)
        rec = _rec_for(result, "svcE")
        # The migration into the critically-full destination must be blocked.
        # (A full destination also raises cluster queue pressure, so the block may
        # come from queue-priority or the destination gate — both are valid.)
        assert rec.target_region != "dst", "must not migrate into a critically-full destination"

    def test_case_e2_destination_gate_blocks_low_spare_migration(self):
        # Isolate the hard destination gate: destination has critically-low spare
        # but does NOT trip cluster queue pressure (kept inactive via unknown spare
        # on the destination so the migration reaches the gate).
        svc = _svc("svcE2", "src", p99=300)
        state = _cluster({
            "src": _region("src", price=200, pct=97, spare=80, services={"svcE2": svc},
                           gpus=[_gpu("h", "src-n0", "src", util=70)]),
            # destination with NO capacity evidence (unverifiable) — the gate must block.
            "dst": _region("dst", price=40, pct=8, spare=None, with_node=False),
        })
        result = _run(state)
        rec = _rec_for(result, "svcE2")
        assert rec.target_region != "dst"
        assert any("destination_unsafe" in r["reject_reason"] for r in result.rejected)

    def test_case_f_safe_migration_may_be_recommended(self):
        svc = _svc("svcF", "src", p99=400)  # huge headroom, no SLA-risk active
        state = _cluster({
            "src": _region("src", price=300, pct=99, spare=90, services={"svcF": svc},
                           gpus=[_gpu("j", "src-n0", "src", util=55)]),
            "dst": _region("dst", price=30, pct=5, spare=80,
                           gpus=[_gpu("k", "dst-n0", "dst", util=35)]),
        })
        rec = _rec_for(_run(state), "svcF")
        assert rec.action_type == ActionType.CHOOSE_CHEAPER_REGION.value
        assert rec.target_region == "dst"

    def test_case_g_missing_destination_telemetry_blocks_migration(self):
        svc = _svc("svcG", "src", p99=400)
        state = _cluster({
            "src": _region("src", price=300, pct=99, spare=80, services={"svcG": svc},
                           gpus=[_gpu("l", "src-n0", "src", util=60)]),
            "dst": _region("dst", price=30, pct=5, spare=None, with_node=False),  # no capacity evidence
        }, is_partial=True, missing=["dst.kubernetes"])
        result = _run(state)
        rec = _rec_for(result, "svcG")
        assert rec.target_region != "dst", "must not migrate when destination telemetry is unverifiable"
        assert any("destination_unsafe" in r["reject_reason"] for r in result.rejected)


# ---------------------------------------------------------------------------
# Full-vector consumption (not just the binding label)
# ---------------------------------------------------------------------------

class TestScoreVectorConsumed:
    def _util_state(self, temp):
        gpus = [_gpu("g1", "r-n0", "r", temp=temp, util=12),
                _gpu("g2", "r-n0", "r", temp=temp, util=12)]
        svc = _svc("svc", "r")
        return _cluster({"r": _region("r", temp=temp, spare=85, services={"svc": svc}, gpus=gpus)})

    def test_secondary_thermal_changes_the_action(self):
        """Same utilization-bound scenario; only the secondary thermal score differs.

        Cool (45C, thermal inactive) → CONSOLIDATE is allowed.
        Warm (80C, thermal active secondary) → CONSOLIDATE is suppressed.
        Proves the engine consumes the secondary score, not just the binding label.
        """
        cool = _run(self._util_state(45))
        warm = _run(self._util_state(80))
        cool_actions = {r.action_type for r in cool.recommendations}
        warm_actions = {r.action_type for r in warm.recommendations}
        assert ActionType.CONSOLIDATE.value in cool_actions
        assert ActionType.CONSOLIDATE.value not in warm_actions

    def test_disallowed_union_blocks_consolidate_under_secondary_thermal(self):
        result = _run(self._util_state(80))
        assert any(
            r["action"] == ActionType.CONSOLIDATE.value or "consolidate" in r["action"]
            for r in result.rejected
        ) or all(rec.action_type != ActionType.CONSOLIDATE.value
                 for rec in result.recommendations)


# ---------------------------------------------------------------------------
# Portfolio across services + explainability
# ---------------------------------------------------------------------------

class TestPortfolioAndExplainability:
    def test_distinct_services_get_constraint_appropriate_actions(self):
        # One region, two services: one queue-bound, one with huge headroom in a
        # cluster where a cheaper region exists. The recommendation set is a
        # per-service portfolio.
        q_svc = _svc("q-svc", "r", qdepth=90, qwait=1600)
        state = _cluster({
            "r": _region("r", price=200, pct=98, spare=8, services={"q-svc": q_svc},
                         gpus=[_gpu("g", "r-n0", "r", util=92)]),
            "cheap": _region("cheap", price=40, pct=8, spare=70,
                             gpus=[_gpu("h", "cheap-n0", "cheap", util=30)]),
        })
        result = _run(state)
        q_rec = _rec_for(result, "q-svc")
        # Queue-bound service must not be migrated for energy.
        assert q_rec.action_type not in _MIGRATION

    def test_rationale_lists_active_constraints(self):
        gpus = [_gpu("g", "r-n0", "r", temp=92, util=88)]
        svc = _svc("svc", "r")
        state = _cluster({"r": _region("r", temp=92, throttling=True, spare=40,
                                       services={"svc": svc}, gpus=gpus)})
        rec = _rec_for(_run(state), "svc")
        assert "constraint" in rec.rationale.lower()

    def test_rejected_alternatives_have_reasons(self):
        # Energy migration rejected by the destination gate (unverifiable dest) →
        # the rejection is recorded with a reason for observability.
        svc = _svc("svc", "src", p99=300)
        state = _cluster({
            "src": _region("src", price=200, pct=97, spare=80, services={"svc": svc},
                           gpus=[_gpu("h", "src-n0", "src", util=70)]),
            "dst": _region("dst", price=40, pct=8, spare=None, with_node=False),
        })
        result = _run(state)
        assert result.rejected
        assert all("reject_reason" in r and r["reject_reason"] for r in result.rejected)


# ---------------------------------------------------------------------------
# Recommendation-only safety preserved
# ---------------------------------------------------------------------------

class TestSafetyPreserved:
    def test_all_recommendation_only(self):
        svc = _svc("svc", "r", qdepth=90, qwait=1500)
        state = _cluster({"r": _region("r", spare=5, services={"svc": svc},
                                       gpus=[_gpu("g", "r-n0", "r", util=95)])})
        result = _run(state)
        assert all(r.implementation_mode == "recommendation_only" for r in result.recommendations)
