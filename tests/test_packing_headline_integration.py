"""Part F — packing baselines must not be silently replaced by FIFO.

The cluster simulator cannot honestly EXECUTE first-fit / best-fit / FFD as a
closed-loop scheduling policy (see aurelius/benchmarks/packing.py docstring), so
packing primitives are computed as an analysis-only frontier. These tests pin
the honest behaviour required by docs/RESULTS.md §3:

  * a fragmentation/packing scenario with NO computed packing baseline must
    surface the explicit disclaimer `no_packing_baseline_computed_for_this_run`
    — NOT silently fall back to FIFO as if it were the packing baseline;
  * when packing baselines ARE present in the results, headline selection picks
    the strongest packing primitive, never FIFO;
  * the packing primitives + clairvoyant lower bound are still computed for
    fragmentation scenarios (analysis frontier), and the lower bound is never a
    deployable headline.
"""

from __future__ import annotations

from aurelius.benchmarks.packing import analyze_cluster_packing, clairvoyant_lower_bound
from aurelius.benchmarks.per_workload import (
    ScenarioMetadata,
    select_headline_baseline,
)


def _frag_metadata():
    return ScenarioMetadata(
        scenario_name="fragmentation_stranded_capacity",
        primary_workload_type="batch_inference",
        optimization_intent="fragmentation_packing",
        relevant_baselines=("fifo", "first_fit", "best_fit", "first_fit_decreasing"),
        headline_baseline_override=None,
        goodput_unit="tokens",
        sla_slo_type="throughput_only",
        is_telemetry_failsafe=False,
    )


class _KPI:
    def __init__(self, gpd, sla=0, p99=100.0):
        self.sla_safe_goodput_per_infra_dollar = gpd
        self.sla_violations = sla
        self.p99_latency_ms = p99


def test_fragmentation_without_packing_baselines_emits_disclaimer():
    # Only the standard policies are present — no packing primitive was computed.
    results = {"fifo": _KPI(0.1), "constraint_aware": _KPI(0.2)}
    name, rationale = select_headline_baseline(_frag_metadata(), results)
    assert name == "fifo"
    assert rationale == "no_packing_baseline_computed_for_this_run", (
        "fragmentation scenario must NOT silently treat FIFO as the packing "
        "baseline — it must surface the missing-data disclaimer"
    )


def test_fragmentation_with_packing_baselines_picks_strongest_packing():
    results = {
        "fifo": _KPI(0.10),
        "first_fit": _KPI(0.20),
        "best_fit": _KPI(0.25),
        "first_fit_decreasing": _KPI(0.22),
    }
    name, rationale = select_headline_baseline(_frag_metadata(), results)
    assert name == "best_fit"  # strongest packing primitive
    assert name != "fifo"
    assert "packing" in rationale


def test_clairvoyant_lower_bound_is_analysis_only():
    # The clairvoyant bound is never offered to headline selection as a
    # deployable candidate (it is not in the packing_names set there).
    results = {
        "fifo": _KPI(0.10),
        "clairvoyant_lower_bound": _KPI(99.0),  # absurdly strong oracle
    }
    name, _ = select_headline_baseline(_frag_metadata(), results)
    assert name != "clairvoyant_lower_bound"


def test_packing_frontier_computed_for_fragmentation_scenario():
    # The packing frontier (the analysis baseline set) is computable from a real
    # fragmentation cluster state — i.e. the data IS present, just not as a
    # closed-loop policy.
    from aurelius.simulation.cluster.engine import ClusterSimulator
    from aurelius.simulation.cluster.scenarios import load_scenario
    sc = load_scenario("fragmentation_stranded_capacity", seed_override=42)
    sim = ClusterSimulator(sc.config, seed=42)
    state = sim.tick().cluster_state
    analyses = analyze_cluster_packing(state)
    assert analyses, "fragmentation scenario should yield a packing frontier"
    for a in analyses:
        clair = clairvoyant_lower_bound(
            [d for d in [n.gpu_allocated for n in state.regions[a.region].nodes.values()]
             if d and d > 0],
            a.bin_capacity,
        )
        for name in ("first_fit", "best_fit", "first_fit_decreasing"):
            assert a.results[name].bins_used >= clair.bins_used
