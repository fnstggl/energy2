"""Tests for the packing baselines (task spec §1.6)."""

from aurelius.benchmarks.packing import (
    analyze_cluster_packing,
    best_fit,
    clairvoyant_lower_bound,
    first_fit,
    first_fit_decreasing,
    greedy_bin_packing,
)


def test_first_fit_basic():
    r = first_fit([4, 4, 4, 4], 8)
    assert r.bins_used == 2
    assert r.total_demand == 16
    assert r.stranded == 0
    assert r.packing_density == 1.0


def test_best_fit_packs_tightly():
    # best-fit should not do worse than first-fit here.
    items = [5, 4, 3, 3, 1]
    ff = first_fit(items, 8)
    bf = best_fit(items, 8)
    assert bf.bins_used <= ff.bins_used


def test_ffd_no_worse_than_first_fit():
    items = [2, 5, 4, 7, 1, 3, 8]
    ff = first_fit(items, 8)
    ffd = first_fit_decreasing(items, 8)
    assert ffd.bins_used <= ff.bins_used


def test_clairvoyant_is_lower_bound():
    items = [3, 3, 3, 3, 3]  # total 15, cap 8 → floor = 2
    clair = clairvoyant_lower_bound(items, 8)
    assert clair.bins_used == 2
    for heuristic in (first_fit, best_fit, first_fit_decreasing, greedy_bin_packing):
        r = heuristic(items, 8)
        # No online heuristic can beat the clairvoyant floor.
        assert r.bins_used >= clair.bins_used


def test_oversized_items_split_across_bins():
    # A 10-GPU workload on 4-GPU nodes needs at least 3 nodes.
    r = first_fit([10], 4)
    assert r.bins_used == 3
    assert r.total_demand == 10


def test_empty_and_zero_inputs():
    assert first_fit([], 8).bins_used == 0
    assert first_fit([0, 0], 8).bins_used == 0
    assert first_fit([4], 0).bins_used == 0


def test_density_in_unit_range():
    r = greedy_bin_packing([3, 5, 2, 6, 1], 8)
    assert 0.0 < r.packing_density <= 1.0


def test_analyze_cluster_packing_reveals_stranded_node():
    from aurelius.simulation.cluster import load_scenario
    from aurelius.simulation.cluster.engine import ClusterSimulator

    sc = load_scenario("fragmentation_stranded_capacity", seed_override=42)
    sim = ClusterSimulator(sc.config, seed=42)
    sim.run(steps=4)
    analyses = analyze_cluster_packing(sim.get_cluster_state())
    assert analyses, "fragmentation scenario should yield at least one region analysis"
    a = analyses[0]
    # The packing heuristics should never need MORE nodes than are active now.
    for r in a.results.values():
        assert r.bins_used <= a.current_active_nodes
    # And never beat the clairvoyant floor.
    clair = a.results["clairvoyant_lower_bound"].bins_used
    for name, r in a.results.items():
        if name != "clairvoyant_lower_bound":
            assert r.bins_used >= clair


def test_to_dict_shape():
    d = first_fit([4, 4], 8).to_dict()
    assert set(d) >= {
        "heuristic", "bins_used", "stranded_gpus", "packing_density",
        "item_count", "total_demand", "bin_capacity",
    }
