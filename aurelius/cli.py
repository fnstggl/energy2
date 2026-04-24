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


def cmd_backtest(args):
    """Run walk-forward backtest against historical price/carbon CSV files."""
    import pandas as pd
    from .backtesting.engine import BacktestEngine
    from .ingestion.grid_apis.csv_importer import CSVPriceImporter, CSVCarbonImporter
    from .ingestion.job_logs import JobLogIngester
    from .models import OptimizationConfig

    regions = [r.strip() for r in args.regions.split(",")]
    start_ts = pd.Timestamp(args.start, tz="UTC") if args.start else None
    end_ts = pd.Timestamp(args.end, tz="UTC") if args.end else None

    # Load price data
    price_importer = CSVPriceImporter(args.price_source)
    price_df = price_importer.load_all()
    if price_df.empty:
        print(f"ERROR: No price data loaded from {args.price_source}", file=sys.stderr)
        sys.exit(1)

    # Filter to requested regions
    price_df = price_df[price_df["region"].isin(regions)]

    # Load carbon data (optional)
    if args.carbon_source:
        carbon_importer = CSVCarbonImporter(args.carbon_source)
        carbon_df = carbon_importer.load_all()
        if not carbon_df.empty:
            carbon_df = carbon_df[carbon_df["region"].isin(regions)]
    else:
        from .ingestion.grid_apis.base import empty_carbon_df
        carbon_df = empty_carbon_df()

    # Generate synthetic jobs if no job file provided
    if args.jobs_file:
        job_ingester = JobLogIngester()
        jobs = job_ingester.load_from_json(args.jobs_file)
    else:
        from datetime import datetime
        job_ingester = JobLogIngester()
        sim_start = start_ts.to_pydatetime() if start_ts else datetime.utcnow()
        jobs = job_ingester.generate_synthetic(
            start_time=sim_start,
            duration_hours=args.train_days * 24 + args.eval_days * 24,
            num_jobs=args.num_jobs,
            regions=regions,
            seed=42,
        )

    config = OptimizationConfig()

    engine = BacktestEngine(
        method=args.method,
        train_days=args.train_days,
        eval_days=args.eval_days,
        config=config,
    )

    print(f"\nRunning backtest: {args.train_days}d train / {args.eval_days}d eval windows")
    print(f"Price source: {args.price_source}")
    print(f"Regions: {regions}")
    print()

    rounds = engine.run(jobs, price_df, carbon_df, start=start_ts, end=end_ts)

    if not rounds:
        print("No backtest folds produced. Check date range and data coverage.")
        sys.exit(1)

    # Print summary table
    print(f"{'Fold':>4}  {'Eval Start':>20}  {'Jobs':>5}  {'Optimizer $':>12}  {'FIFO $':>12}  {'Savings%':>9}")
    print("-" * 75)
    for r in rounds:
        opt_cost = r.optimizer_metrics.total_energy_cost_usd if r.optimizer_metrics else 0
        fifo_cost = r.baseline_metrics.get("fifo", None)
        fifo_val = fifo_cost.total_energy_cost_usd if fifo_cost else float("nan")
        savings = ((fifo_val - opt_cost) / fifo_val * 100) if fifo_val > 0 else float("nan")
        print(
            f"{r.fold_index:>4}  {str(r.eval_start)[:19]:>20}  "
            f"{len(r.eval_jobs):>5}  ${opt_cost:>11.2f}  ${fifo_val:>11.2f}  "
            f"{savings:>8.1f}%"
        )

    print()
    print(f"Total folds: {len(rounds)}")

    # Save JSON if requested
    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump([r.to_dict() for r in rounds], f, indent=2, default=str)
        print(f"Results saved to: {output_path}")


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

    # Backtest command
    bt_parser = subparsers.add_parser(
        "backtest",
        help="Run leakage-free walk-forward backtest on historical data",
    )
    bt_parser.add_argument(
        "--price-source", required=True,
        help="Path to CSV file with columns: timestamp, region, price_per_mwh",
    )
    bt_parser.add_argument(
        "--carbon-source", default=None,
        help="Path to CSV file with columns: timestamp, region, gco2_per_kwh (optional)",
    )
    bt_parser.add_argument(
        "--jobs-file", default=None,
        help="Path to JSON job log file (generates synthetic jobs if omitted)",
    )
    bt_parser.add_argument(
        "--num-jobs", type=int, default=20,
        help="Number of synthetic jobs to generate per fold if --jobs-file not given",
    )
    bt_parser.add_argument(
        "--start", default=None,
        help="Backtest start date (ISO 8601, e.g. 2024-01-01)",
    )
    bt_parser.add_argument(
        "--end", default=None,
        help="Backtest end date (ISO 8601, e.g. 2024-03-01)",
    )
    bt_parser.add_argument(
        "--regions", default="us-west,us-east,eu-west",
        help="Comma-separated list of regions to include",
    )
    bt_parser.add_argument(
        "--train-days", type=int, default=30,
        help="Training window length in days (default: 30)",
    )
    bt_parser.add_argument(
        "--eval-days", type=int, default=7,
        help="Evaluation window length in days (default: 7)",
    )
    bt_parser.add_argument(
        "--method", default="greedy",
        choices=["greedy", "local_search"],
        help="Optimizer method (default: greedy)",
    )
    bt_parser.add_argument(
        "--output", default=None,
        help="Save results as JSON to this path",
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
    elif args.command == "backtest":
        cmd_backtest(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
