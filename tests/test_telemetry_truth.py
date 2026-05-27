"""Mission 1 — telemetry confidence / partial-state truth.

These tests prove the simulator no longer masks degraded telemetry as perfect:
the canonical ClusterState now derives provenance confidence + is_partial from
the simulator's own per-subsystem tiers, and degraded telemetry forces KEEP.
"""

import pytest

from aurelius.constraints import ConstraintAwareEngine
from aurelius.constraints.classifier import ConstraintClassifier
from aurelius.simulation.cluster import load_scenario
from aurelius.simulation.cluster.engine import ClusterSimulator

CLEAN = "energy_price_arbitrage_multiregion"
DEGRADED = [
    "degraded_topology_telemetry",
    "partial_utilization_telemetry",
    "low_confidence_energy_telemetry",
]


def _state(scn, steps=6, seed=42):
    sc = load_scenario(scn, seed_override=seed)
    sim = ClusterSimulator(sc.config, seed=seed)
    sim.run(steps=steps)
    return sim.get_cluster_state()


def test_clean_scenario_is_high_confidence_not_partial():
    st = _state(CLEAN)
    assert st.provenance.confidence == "high"
    assert st.is_partial is False
    assert st.missing_sources == []


@pytest.mark.parametrize("scn", DEGRADED)
def test_degraded_scenario_is_not_high_confidence(scn):
    st = _state(scn)
    assert st.provenance.confidence != "high", (
        f"{scn} must NOT report perfect telemetry"
    )


@pytest.mark.parametrize("scn", DEGRADED)
def test_degraded_scenario_marks_partial_with_sources(scn):
    st = _state(scn)
    assert st.is_partial is True
    assert st.missing_sources, "degraded telemetry must populate missing_sources"


@pytest.mark.parametrize("scn", DEGRADED)
def test_degraded_telemetry_forces_keep(scn):
    st = _state(scn)
    res = ConstraintAwareEngine().run(st)
    assert res.actionable_count == 0, (
        f"{scn}: degraded telemetry must force KEEP (no risky action)"
    )
    assert all(r.is_noop for r in res.recommendations)


def test_clean_scenario_still_acts():
    # The telemetry-trust gate must NOT suppress legitimate clean scenarios.
    st = _state("thermal_hotspot_mixed_cluster")
    assert st.provenance.confidence == "high"
    res = ConstraintAwareEngine().run(st)
    assert res.actionable_count >= 1


def test_low_coverage_but_trustworthy_scenario_not_gated_by_telemetry():
    # rack_density / fragmentation have low *classifier* confidence (low coverage)
    # but are NOT partial — they must not be force-KEEP'd by the telemetry gate.
    st = _state("fragmentation_stranded_capacity")
    assert st.is_partial is False
    assert st.provenance.confidence == "high"


def test_missing_telemetry_never_becomes_zero():
    # A degraded region must lower confidence, never fabricate a zero metric that
    # reads as a safe opportunity. Confidence drops; it does not invert to "great".
    clean = _state(CLEAN)
    degraded = _state("low_confidence_energy_telemetry")
    clean_a = ConstraintClassifier().assess(clean)
    deg_a = ConstraintClassifier().assess(degraded)
    assert deg_a.confidence < clean_a.confidence


def test_telemetry_truth_is_deterministic():
    a = _state("degraded_topology_telemetry")
    b = _state("degraded_topology_telemetry")
    assert a.provenance.confidence == b.provenance.confidence
    assert a.is_partial == b.is_partial
