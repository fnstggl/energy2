"""Simulation modules for Aurelius."""

# Lazy imports: these depend on pandas/database which may not be present in
# all environments (e.g. the constraint-aware test environment). Import them
# only when actually used, so that aurelius.simulation.cluster (which has no
# such dependency) can always be imported.

try:
    from .compare import ScenarioComparator
    from .metrics import MetricsCalculator
    from .replay import SimulationReplay
    __all__ = ["SimulationReplay", "ScenarioComparator", "MetricsCalculator"]
except ImportError:
    # pandas / database layer not available in this environment
    __all__ = []
