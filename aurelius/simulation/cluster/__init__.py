"""Synthetic cluster simulator for constraint-aware Aurelius validation.

This package provides a discrete-event cluster simulator that exercises the
same connector interfaces (Prometheus, Kubernetes, topology, energy) as real
customer integrations. Aurelius cannot distinguish simulator from real at the
connector boundary.

All simulator outputs are marked is_sandbox=True and excluded from any
economic/SLA claim.

Usage:
    from aurelius.simulation.cluster import ClusterSimulator, load_scenario

    config = load_scenario("energy_price_arbitrage_multiregion")
    sim = ClusterSimulator(config, seed=42)
    ticks = sim.run(steps=24)  # 24 hourly ticks
    cluster_state = sim.get_cluster_state()
"""

from .engine import ClusterSimulator, SimulatorTick
from .model import (
    SimCluster,
    SimGPU,
    SimNode,
    SimQueue,
    SimRegion,
    SimulatorConfig,
    SimWorkload,
)
from .scenarios import ScenarioConfig, list_scenarios, load_scenario
from .topology_model import (
    GPUFabricState,
    NodeFabricState,
    WorkloadTopologyState,
)

__all__ = [
    "ClusterSimulator",
    "SimulatorTick",
    "SimCluster",
    "SimGPU",
    "SimNode",
    "SimQueue",
    "SimRegion",
    "SimWorkload",
    "SimulatorConfig",
    "load_scenario",
    "list_scenarios",
    "ScenarioConfig",
    "GPUFabricState",
    "NodeFabricState",
    "WorkloadTopologyState",
]
