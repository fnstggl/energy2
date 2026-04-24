"""Command-line interface for Aurelius.

Usage:
    python -m aurelius.cli simulate [options]
    python -m aurelius.cli generate-data [options]
    python -m aurelius.cli robustness-test [options]
    python -m aurelius.cli show-schema

Examples:
    # Run a simulation with defaults
    python -m aurelius.cli simulate

    # Run with custom parameters
    python -m aurelius.cli simulate --jobs 100 --hours 72 --method local_search

    # Generate synthetic data files
    python -m aurelius.cli generate-data --output ./data/

    # Run robustness test (20 runs by default)
    python -m aurelius.cli robustness-test --runs 20 --output report.json
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("aurelius")


def cmd_simulate(args):
    """Run a simulation."""
    from .simulation.replay import SimulationReplay, SimulationConfig
    from .models import OptimizationConfig

    # Parse regions
    regions = [r.strip() for r in args.regions.split(",")]

    # Create configuration
    opt_config = OptimizationConfig(
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        min_power_fraction=args.min_power,
    )

    sim_config = SimulationConfig(
        start_time=datetime.utcnow(),
        duration_hours=args.hours,
        regions=regions,
        num_jobs=args.jobs,
        optimization_method=args.method,
        optimization_config=opt_config,
        price_scenario=args.price_scenario,
        carbon_scenario=args.carbon_scenario,
        random_seed=args.seed,
        save_to_db=not args.no_db,
    )

    # Run simulation
    replay = SimulationReplay()
    results = replay.run(sim_config)

    # Print summary with dual baseline comparison
    metrics = results.get('metrics', {})
    baselines = metrics.get('baselines', {})
    optimized = metrics.get('optimized', {})
    savings_fifo = metrics.get('savings_vs_fifo', {})
    savings_peak = metrics.get('savings_vs_peak_blind', {})

    print("\n" + "=" * 70)
    print("AURELIUS SIMULATION COMPLETE")
    print("=" * 70)
    print(f"Run ID: {results['run_id']}")
    print(f"Jobs Scheduled: {results['summary']['jobs_scheduled']}")
    print()

    print("-" * 70)
    print("BASELINE SCENARIOS")
    print("-" * 70)
    fifo = baselines.get('fifo', {})
    peak = baselines.get('peak_blind', {})
    print()
    print("FIFO BASELINE (jobs run in submission order, no optimization):")
    print(f"  Energy Cost:      ${fifo.get('energy_cost', 0):>12,.2f}")
    print(f"  Compute Cost:     ${fifo.get('compute_cost', 0):>12,.2f}")
    print(f"  Carbon:           {fifo.get('carbon_kg', 0):>13,.2f} kg CO2")
    print()
    print("PEAK-BLIND ASAP BASELINE (jobs run immediately, even during peaks):")
    print(f"  Energy Cost:      ${peak.get('energy_cost', 0):>12,.2f}")
    print(f"  Compute Cost:     ${peak.get('compute_cost', 0):>12,.2f}")
    print(f"  Carbon:           {peak.get('carbon_kg', 0):>13,.2f} kg CO2")
    print()

    print("-" * 70)
    print("OPTIMIZED SCHEDULE")
    print("-" * 70)
    print(f"  Energy Cost:      ${optimized.get('energy_cost', 0):>12,.2f}")
    print(f"  Compute Cost:     ${optimized.get('compute_cost', 0):>12,.2f}")
    print(f"  Carbon:           {optimized.get('carbon_kg', 0):>13,.2f} kg CO2")
    print(f"  Jobs Throttled:   {optimized.get('jobs_throttled', 0):>13}")
    print(f"  Jobs Shifted:     {optimized.get('jobs_shifted', 0):>13}")
    print()

    print("-" * 70)
    print("SAVINGS VS FIFO BASELINE")
    print("-" * 70)
    print(f"  Energy Cost:      ${savings_fifo.get('energy_cost_savings_dollars', 0):>12,.2f}  ({savings_fifo.get('energy_cost_savings_pct', 0):>6.1f}%)")
    print(f"  Compute Cost:     ${savings_fifo.get('compute_cost_savings_dollars', 0):>12,.2f}  ({savings_fifo.get('compute_cost_savings_pct', 0):>6.1f}%)")
    print(f"  Carbon:           {savings_fifo.get('carbon_savings_kg', 0):>13,.2f} kg ({savings_fifo.get('carbon_savings_pct', 0):>6.1f}%)")
    print()

    print("-" * 70)
    print("SAVINGS VS PEAK-BLIND BASELINE")
    print("-" * 70)
    print(f"  Energy Cost:      ${savings_peak.get('energy_cost_savings_dollars', 0):>12,.2f}  ({savings_peak.get('energy_cost_savings_pct', 0):>6.1f}%)")
    print(f"  Compute Cost:     ${savings_peak.get('compute_cost_savings_dollars', 0):>12,.2f}  ({savings_peak.get('compute_cost_savings_pct', 0):>6.1f}%)")
    print(f"  Carbon:           {savings_peak.get('carbon_savings_kg', 0):>13,.2f} kg ({savings_peak.get('carbon_savings_pct', 0):>6.1f}%)")
    print()

    print("-" * 70)
    print("REGION DISTRIBUTION (Optimized)")
    print("-" * 70)
    for region, count in sorted(optimized.get('region_distribution', {}).items()):
        print(f"  {region}: {count} jobs")
    print("=" * 70)

    # Save to file if requested
    if args.output:
        output_path = Path(args.output)
        replay.save_results_to_file(results, output_path)
        print(f"\nResults saved to: {output_path}")


def cmd_generate_data(args):
    """Generate synthetic data files."""
    from .ingestion.energy_prices import EnergyPriceIngester
    from .ingestion.job_logs import JobLogIngester
    from .forecasting.baseline import generate_carbon_scenario

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    regions = [r.strip() for r in args.regions.split(",")]
    start_time = datetime.utcnow()

    # Generate energy prices
    price_ingester = EnergyPriceIngester()
    prices = price_ingester.generate_synthetic(
        start_time=start_time,
        hours=args.hours,
        regions=regions,
        seed=args.seed,
    )
    price_file = output_dir / "energy_prices.csv"
    price_ingester.save_to_csv(prices, price_file)
    print(f"Generated {len(prices)} price records -> {price_file}")

    # Generate carbon data
    carbon_data = generate_carbon_scenario(
        start_time=start_time,
        hours=args.hours,
        regions=regions,
        seed=args.seed,
    )
    carbon_file = output_dir / "carbon_intensity.csv"
    import csv
    with open(carbon_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "region", "gco2_per_kwh"])
        writer.writeheader()
        for c in carbon_data:
            writer.writerow({
                "timestamp": c.timestamp.isoformat(),
                "region": c.region,
                "gco2_per_kwh": c.gco2_per_kwh,
            })
    print(f"Generated {len(carbon_data)} carbon records -> {carbon_file}")

    # Generate jobs
    job_ingester = JobLogIngester()
    jobs = job_ingester.generate_synthetic(
        start_time=start_time,
        duration_hours=args.hours,
        num_jobs=args.jobs,
        regions=regions,
        seed=args.seed,
    )
    job_file = output_dir / "jobs.json"
    job_ingester.save_to_json(jobs, job_file)
    print(f"Generated {len(jobs)} jobs -> {job_file}")

    print(f"\nData generation complete. Files saved to: {output_dir}")


def cmd_show_schema(args):
    """Show database schema."""
    from .database import print_schema
    print_schema()


def cmd_robustness_test(args):
    """Run robustness test harness."""
    from .validation.robustness import (
        RobustnessTestHarness,
        format_cli_report,
        save_report_json,
    )

    # Suppress verbose logging during test runs
    logging.getLogger("aurelius").setLevel(logging.WARNING)

    regions = [r.strip() for r in args.regions.split(",")]

    harness = RobustnessTestHarness(
        num_jobs=args.jobs,
        duration_hours=args.hours,
        regions=regions,
        optimization_method=args.method,
        price_scenario=args.price_scenario,
        carbon_scenario=args.carbon_scenario,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
    )

    print(f"\nRunning robustness test: {args.runs} simulations...")
    print(f"Configuration: {args.jobs} jobs, {args.hours}h duration, method={args.method}")
    print()

    report = harness.run(num_runs=args.runs, base_seed=args.base_seed)

    # Print CLI summary
    print(format_cli_report(report))

    # Save JSON report if requested
    if args.output:
        output_path = Path(args.output)
        save_report_json(report, output_path)
        print(f"JSON report saved to: {output_path}")

    # Exit with error code if unstable
    if not report.is_stable:
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Aurelius - Predictive control for energy-constrained batch compute",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Simulate command
    sim_parser = subparsers.add_parser("simulate", help="Run a simulation")
    sim_parser.add_argument(
        "--jobs", type=int, default=50,
        help="Number of jobs to simulate (default: 50)"
    )
    sim_parser.add_argument(
        "--hours", type=int, default=168,
        help="Simulation duration in hours (default: 168 = 1 week)"
    )
    sim_parser.add_argument(
        "--regions", type=str, default="us-west,us-east,eu-west",
        help="Comma-separated list of regions"
    )
    sim_parser.add_argument(
        "--method", type=str, default="greedy",
        choices=["greedy", "local_search", "milp"],
        help="Optimization method (default: greedy)"
    )
    sim_parser.add_argument(
        "--alpha", type=float, default=1.0,
        help="Weight for energy cost objective (default: 1.0)"
    )
    sim_parser.add_argument(
        "--beta", type=float, default=0.3,
        help="Weight for carbon cost objective (default: 0.3)"
    )
    sim_parser.add_argument(
        "--gamma", type=float, default=0.05,
        help="Weight for risk penalty (default: 0.05)"
    )
    sim_parser.add_argument(
        "--min-power", type=float, default=0.5,
        help="Minimum power throttle fraction (default: 0.5)"
    )
    sim_parser.add_argument(
        "--price-scenario", type=str, default="normal",
        choices=["normal", "volatile", "low", "high", "peak_valley"],
        help="Price scenario for synthetic data"
    )
    sim_parser.add_argument(
        "--carbon-scenario", type=str, default="normal",
        choices=["normal", "clean", "dirty", "variable"],
        help="Carbon scenario for synthetic data"
    )
    sim_parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    sim_parser.add_argument(
        "--output", type=str,
        help="Output file path for results JSON"
    )
    sim_parser.add_argument(
        "--no-db", action="store_true",
        help="Don't save results to database"
    )

    # Generate data command
    gen_parser = subparsers.add_parser("generate-data", help="Generate synthetic data files")
    gen_parser.add_argument(
        "--output", type=str, default="./data/processed",
        help="Output directory"
    )
    gen_parser.add_argument(
        "--hours", type=int, default=168,
        help="Hours of data to generate"
    )
    gen_parser.add_argument(
        "--jobs", type=int, default=100,
        help="Number of jobs to generate"
    )
    gen_parser.add_argument(
        "--regions", type=str, default="us-west,us-east,eu-west",
        help="Comma-separated list of regions"
    )
    gen_parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )

    # Show schema command
    schema_parser = subparsers.add_parser("show-schema", help="Show database schema")

    # Robustness test command
    robust_parser = subparsers.add_parser(
        "robustness-test",
        help="Run robustness test harness to validate optimizer stability"
    )
    robust_parser.add_argument(
        "--runs", type=int, default=20,
        help="Number of simulation runs (default: 20)"
    )
    robust_parser.add_argument(
        "--base-seed", type=int, default=1000,
        help="Starting random seed (default: 1000)"
    )
    robust_parser.add_argument(
        "--jobs", type=int, default=50,
        help="Number of jobs per simulation (default: 50)"
    )
    robust_parser.add_argument(
        "--hours", type=int, default=72,
        help="Simulation duration in hours (default: 72)"
    )
    robust_parser.add_argument(
        "--regions", type=str, default="us-west,us-east,eu-west",
        help="Comma-separated list of regions"
    )
    robust_parser.add_argument(
        "--method", type=str, default="greedy",
        choices=["greedy", "local_search", "milp"],
        help="Optimization method (default: greedy)"
    )
    robust_parser.add_argument(
        "--alpha", type=float, default=1.0,
        help="Weight for energy cost objective (default: 1.0)"
    )
    robust_parser.add_argument(
        "--beta", type=float, default=0.3,
        help="Weight for carbon cost objective (default: 0.3)"
    )
    robust_parser.add_argument(
        "--gamma", type=float, default=0.05,
        help="Weight for risk penalty (default: 0.05)"
    )
    robust_parser.add_argument(
        "--price-scenario", type=str, default="normal",
        choices=["normal", "volatile", "low", "high", "peak_valley"],
        help="Price scenario for synthetic data"
    )
    robust_parser.add_argument(
        "--carbon-scenario", type=str, default="normal",
        choices=["normal", "clean", "dirty", "variable"],
        help="Carbon scenario for synthetic data"
    )
    robust_parser.add_argument(
        "--output", type=str,
        help="Output file path for JSON report"
    )

    # Parse arguments
    args = parser.parse_args()

    if args.command == "simulate":
        cmd_simulate(args)
    elif args.command == "generate-data":
        cmd_generate_data(args)
    elif args.command == "show-schema":
        cmd_show_schema(args)
    elif args.command == "robustness-test":
        cmd_robustness_test(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
