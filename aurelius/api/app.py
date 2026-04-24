"""FastAPI application for Aurelius.

Provides a minimal REST API for running simulations.

Endpoints:
- POST /simulate: Run a simulation with custom parameters
- GET /health: Health check
- GET /simulations: List past simulations (from database)
- GET /simulations/{run_id}: Get specific simulation results

No authentication, no pagination, no streaming.
This is a research-grade API for running simulations.
"""

from datetime import datetime
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import logging

from ..models import Job, OptimizationConfig
from ..simulation.replay import SimulationReplay, SimulationConfig
from ..database import get_db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Aurelius",
    description="Predictive control layer for energy-constrained batch compute",
    version="1.0.0",
)


# Request/Response models
class JobInput(BaseModel):
    """Job input for simulation."""
    job_id: str
    submit_time: datetime
    runtime_hours: float = Field(gt=0)
    deadline: datetime
    power_kw: float = Field(gt=0)
    earliest_start: datetime
    region_options: list[str]
    priority: int = 1


class SimulationRequest(BaseModel):
    """Request body for simulation endpoint."""
    jobs: Optional[list[JobInput]] = None
    start_time: Optional[datetime] = None
    duration_hours: int = Field(default=168, gt=0, le=720)
    num_jobs: int = Field(default=50, gt=0, le=1000)
    regions: list[str] = Field(default=["us-west", "us-east", "eu-west"])
    optimization_method: str = Field(default="greedy")
    alpha: float = Field(default=1.0, ge=0)
    beta: float = Field(default=0.1, ge=0)
    gamma: float = Field(default=0.05, ge=0)
    price_scenario: str = Field(default="normal")
    carbon_scenario: str = Field(default="normal")
    random_seed: Optional[int] = 42


class SimulationResponse(BaseModel):
    """Response from simulation endpoint."""
    run_id: str
    created_at: str
    baseline_cost: float
    optimized_cost: float
    cost_savings_pct: float
    baseline_carbon_kg: float
    optimized_carbon_kg: float
    carbon_savings_pct: float
    jobs_scheduled: int
    summary: dict
    config: dict


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    database_connected: bool


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    db = get_db()
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        database_connected=db.is_connected,
    )


@app.post("/simulate", response_model=SimulationResponse)
async def run_simulation(request: SimulationRequest):
    """Run a simulation.

    If jobs are provided, uses those. Otherwise generates synthetic jobs.

    Returns simulation results with baseline vs optimized comparison.
    """
    try:
        # Convert job inputs to Job objects if provided
        jobs = None
        if request.jobs:
            jobs = [
                Job(
                    job_id=j.job_id,
                    submit_time=j.submit_time,
                    runtime_hours=j.runtime_hours,
                    deadline=j.deadline,
                    power_kw=j.power_kw,
                    earliest_start=j.earliest_start,
                    region_options=j.region_options,
                    priority=j.priority,
                )
                for j in request.jobs
            ]

        # Create configuration
        opt_config = OptimizationConfig(
            alpha=request.alpha,
            beta=request.beta,
            gamma=request.gamma,
        )

        sim_config = SimulationConfig(
            start_time=request.start_time or datetime.utcnow(),
            duration_hours=request.duration_hours,
            regions=request.regions,
            num_jobs=request.num_jobs,
            optimization_method=request.optimization_method,
            optimization_config=opt_config,
            price_scenario=request.price_scenario,
            carbon_scenario=request.carbon_scenario,
            random_seed=request.random_seed,
            save_to_db=True,
        )

        # Run simulation
        replay = SimulationReplay()
        results = replay.run(sim_config, jobs=jobs)

        # Format response
        return SimulationResponse(
            run_id=results["run_id"],
            created_at=results["created_at"],
            baseline_cost=results["summary"]["baseline_cost"],
            optimized_cost=results["summary"]["optimized_cost"],
            cost_savings_pct=results["summary"]["cost_savings_pct"],
            baseline_carbon_kg=results["summary"]["baseline_carbon_kg"],
            optimized_carbon_kg=results["summary"]["optimized_carbon_kg"],
            carbon_savings_pct=results["summary"]["carbon_savings_pct"],
            jobs_scheduled=results["summary"]["jobs_scheduled"],
            summary=results["metrics"],
            config=results["simulation_config"],
        )

    except Exception as e:
        logger.exception("Simulation failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/simulations")
async def list_simulations(limit: int = 100):
    """List past simulation runs from database.

    Args:
        limit: Maximum number of results (default 100)

    Returns:
        List of simulation summaries
    """
    db = get_db()
    if not db.is_connected:
        return {"simulations": [], "message": "Database not connected"}

    simulations = db.get_simulations(limit=limit)
    return {"simulations": simulations}


@app.get("/simulations/{run_id}")
async def get_simulation(run_id: str):
    """Get a specific simulation by run_id.

    Args:
        run_id: The simulation run ID

    Returns:
        Simulation details
    """
    db = get_db()
    if not db.is_connected:
        raise HTTPException(status_code=503, detail="Database not connected")

    simulations = db.get_simulations(run_id=run_id)
    if not simulations:
        raise HTTPException(status_code=404, detail="Simulation not found")

    return simulations[0]


# For running with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
