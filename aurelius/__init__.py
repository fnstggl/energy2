"""Aurelius - Predictive Control Layer for Energy-Constrained Batch Compute.

Aurelius v1 is a comprehensive simulator + optimizer that proves the economic
value of foresight in scheduling batch compute workloads.

Key capabilities:
- Time shifting: delay/advance jobs within slack windows
- Power throttling: stretch jobs to avoid price peaks
- Multi-region routing: simulated region selection
- Carbon-aware weighting: secondary optimization objective
- Risk-aware penalties: account for forecast uncertainty

Example usage:

    from aurelius.simulation.replay import SimulationReplay, SimulationConfig
    from aurelius.models import OptimizationConfig

    # Configure optimization weights
    opt_config = OptimizationConfig(
        alpha=1.0,   # energy cost weight
        beta=0.1,    # carbon cost weight
        gamma=0.05,  # risk penalty weight
    )

    # Configure simulation
    sim_config = SimulationConfig(
        num_jobs=50,
        duration_hours=168,  # 1 week
        optimization_config=opt_config,
    )

    # Run simulation
    replay = SimulationReplay()
    results = replay.run(sim_config)

    # Results include baseline vs optimized comparison
    print(f"Cost savings: {results['summary']['cost_savings_pct']:.1f}%")

CLI usage:

    # Run a simulation
    python -m aurelius.cli simulate --jobs 100 --method local_search

    # Generate synthetic data
    python -m aurelius.cli generate-data --output ./data/

API usage:

    # Start the API server
    uvicorn aurelius.api.app:app --host 0.0.0.0 --port 8000

    # POST /simulate with job batch
    # GET /simulations to list past runs
"""

__version__ = "1.0.0"

from .models import (
    CarbonIntensity,
    EnergyPrice,
    Job,
    OptimizationConfig,
    ScheduleDecision,
    SimulationResult,
)

__all__ = [
    "Job",
    "EnergyPrice",
    "CarbonIntensity",
    "ScheduleDecision",
    "SimulationResult",
    "OptimizationConfig",
]
