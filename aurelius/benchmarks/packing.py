"""Bin-packing baselines for utilization / fragmentation scenarios (task spec §1.6).

These are the classic packing heuristics the spec requires as comparison
baselines: first-fit, best-fit, first-fit-decreasing, and a clairvoyant
(optimal) lower bound that is **analysis-only — never a deployable comparison**.

Why analysis-only rather than a closed-loop scheduling policy: the cluster
simulator has no primitive to relocate an arbitrary workload onto an arbitrary
node (its action vocabulary is add_replica / spread / consolidate / migrate-
region). Faithfully *executing* first-fit/best-fit would require fabricating a
relocation engine the simulator cannot model honestly. Instead we compute, from
the observed cluster state, the active-node count and stranded-GPU count each
heuristic *would* achieve — the packing frontier. constraint_aware's
consolidation can then be measured against that frontier without fabricating an
artificial win.

All functions are pure and deterministic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass
class PackingResult:
    """Result of packing items into capacity-`bin_capacity` bins."""
    heuristic: str
    bins_used: int
    bin_loads: list[int]
    item_count: int
    total_demand: int
    bin_capacity: int

    @property
    def stranded(self) -> int:
        """GPUs sitting idle inside the bins that were opened."""
        return self.bins_used * self.bin_capacity - self.total_demand

    @property
    def packing_density(self) -> float:
        cap = self.bins_used * self.bin_capacity
        return self.total_demand / cap if cap > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "heuristic": self.heuristic,
            "bins_used": self.bins_used,
            "stranded_gpus": self.stranded,
            "packing_density": round(self.packing_density, 4),
            "item_count": self.item_count,
            "total_demand": self.total_demand,
            "bin_capacity": self.bin_capacity,
        }


def _pack(items: list[int], bin_capacity: int, heuristic: str) -> PackingResult:
    """Pack `items` (each ≤ bin_capacity) into bins using the given heuristic.

    Items larger than bin_capacity are split across whole bins (a multi-node
    workload), which matches GPU demand that spans nodes.
    """
    items = [i for i in items if i > 0]
    total = sum(items)
    if bin_capacity <= 0:
        return PackingResult(heuristic, 0, [], len(items), total, bin_capacity)

    # Split oversized items into bin_capacity chunks (multi-node workloads).
    chunks: list[int] = []
    for it in items:
        while it > bin_capacity:
            chunks.append(bin_capacity)
            it -= bin_capacity
        if it > 0:
            chunks.append(it)

    if heuristic == "first_fit_decreasing":
        chunks = sorted(chunks, reverse=True)
    elif heuristic == "greedy_bin_packing":
        # Greedy = FFD by another name in 1D; keep distinct ordering (largest first).
        chunks = sorted(chunks, reverse=True)

    bins: list[int] = []  # remaining capacity per open bin
    loads: list[int] = []  # used capacity per open bin

    for size in chunks:
        if heuristic == "best_fit":
            # Tightest bin that still fits.
            best = -1
            best_rem = bin_capacity + 1
            for idx, rem in enumerate(bins):
                if rem >= size and rem < best_rem:
                    best, best_rem = idx, rem
            if best >= 0:
                bins[best] -= size
                loads[best] += size
            else:
                bins.append(bin_capacity - size)
                loads.append(size)
        else:
            # first_fit / first_fit_decreasing / greedy: first bin that fits.
            placed = False
            for idx, rem in enumerate(bins):
                if rem >= size:
                    bins[idx] -= size
                    loads[idx] += size
                    placed = True
                    break
            if not placed:
                bins.append(bin_capacity - size)
                loads.append(size)

    return PackingResult(
        heuristic=heuristic,
        bins_used=len(loads),
        bin_loads=loads,
        item_count=len(items),
        total_demand=total,
        bin_capacity=bin_capacity,
    )


def first_fit(items: list[int], bin_capacity: int) -> PackingResult:
    return _pack(items, bin_capacity, "first_fit")


def best_fit(items: list[int], bin_capacity: int) -> PackingResult:
    return _pack(items, bin_capacity, "best_fit")


def first_fit_decreasing(items: list[int], bin_capacity: int) -> PackingResult:
    return _pack(items, bin_capacity, "first_fit_decreasing")


def greedy_bin_packing(items: list[int], bin_capacity: int) -> PackingResult:
    return _pack(items, bin_capacity, "greedy_bin_packing")


def clairvoyant_lower_bound(items: list[int], bin_capacity: int) -> PackingResult:
    """Optimal (clairvoyant) lower bound on bins needed — ANALYSIS ONLY.

    This is the information-theoretic floor ceil(total_demand / bin_capacity); a
    real online scheduler can never guarantee it. Per the spec it must NEVER be
    used as a deployable comparison — only to bound how much room a heuristic has
    left.
    """
    items = [i for i in items if i > 0]
    total = sum(items)
    if bin_capacity <= 0:
        bins = 0
    else:
        bins = math.ceil(total / bin_capacity)
    loads = [bin_capacity] * (total // bin_capacity) if bin_capacity > 0 else []
    if bin_capacity > 0 and total % bin_capacity:
        loads.append(total % bin_capacity)
    return PackingResult(
        heuristic="clairvoyant_lower_bound",
        bins_used=bins,
        bin_loads=loads,
        item_count=len(items),
        total_demand=total,
        bin_capacity=bin_capacity,
    )


@dataclass
class ClusterPackingAnalysis:
    """Packing frontier for an observed cluster state (one region)."""
    region: str
    bin_capacity: int
    nodes_available: int
    current_active_nodes: int
    results: dict[str, PackingResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "region": self.region,
            "bin_capacity": self.bin_capacity,
            "nodes_available": self.nodes_available,
            "current_active_nodes": self.current_active_nodes,
            "results": {k: v.to_dict() for k, v in self.results.items()},
        }


def analyze_cluster_packing(state: Any) -> list[ClusterPackingAnalysis]:
    """Compute the packing frontier per region from a canonical ClusterState.

    Items = per-node allocated-GPU demand (so a service spanning nodes keeps its
    multi-node footprint); bins = uniform node GPU capacity. Reports how many
    nodes each heuristic would light up vs. the current placement and the
    clairvoyant floor — i.e. how much fragmentation is recoverable.
    """
    analyses: list[ClusterPackingAnalysis] = []
    for region_id, region in getattr(state, "regions", {}).items():
        nodes = list(getattr(region, "nodes", {}).values())
        if not nodes:
            continue
        caps = [int(n.gpu_capacity) for n in nodes if getattr(n, "gpu_capacity", None)]
        if not caps:
            continue
        bin_capacity = max(caps)  # uniform node size assumption (largest node)
        demands = [
            int(getattr(n, "gpu_allocated", 0) or 0) for n in nodes
        ]
        demands = [d for d in demands if d > 0]
        current_active = sum(1 for d in demands)
        results = {
            "first_fit": first_fit(demands, bin_capacity),
            "best_fit": best_fit(demands, bin_capacity),
            "first_fit_decreasing": first_fit_decreasing(demands, bin_capacity),
            "greedy_bin_packing": greedy_bin_packing(demands, bin_capacity),
            "clairvoyant_lower_bound": clairvoyant_lower_bound(demands, bin_capacity),
        }
        analyses.append(ClusterPackingAnalysis(
            region=region_id,
            bin_capacity=bin_capacity,
            nodes_available=len(nodes),
            current_active_nodes=current_active,
            results=results,
        ))
    return analyses
