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


def _load_price_df(args, regions):
    """Resolve --price-provider / --price-file into a canonical price DataFrame."""
    import pandas as pd
    from .ingestion.grid_apis.market_registry import UnsupportedMarketPriceError

    provider = args.price_provider
    start_ts = pd.Timestamp(args.start, tz="UTC") if args.start else None
    end_ts = pd.Timestamp(args.end, tz="UTC") if args.end else None

    if provider == "csv":
        if not args.price_file:
            print("ERROR: --price-file is required when --price-provider=csv", file=sys.stderr)
            sys.exit(1)
        from .ingestion.grid_apis.csv_importer import CSVPriceImporter
        importer = CSVPriceImporter(args.price_file)
        df = importer.load_all()
        if df.empty:
            print(f"ERROR: No price data loaded from {args.price_file}", file=sys.stderr)
            sys.exit(1)
        return df[df["region"].isin(regions)]

    if provider == "caiso":
        from .ingestion.grid_apis.caiso import CAISOPriceProvider
        for region in regions:
            if region not in ("us-west",):
                print(f"ERROR: CAISO provider only supports us-west (got '{region}')", file=sys.stderr)
                sys.exit(1)
        p = CAISOPriceProvider()
        start_dt = start_ts.to_pydatetime() if start_ts else None
        end_dt = end_ts.to_pydatetime() if end_ts else None
        if not start_dt or not end_dt:
            print("ERROR: --start and --end are required for live provider caiso", file=sys.stderr)
            sys.exit(1)
        dfs = []
        for region in regions:
            dfs.append(p.fetch_prices(region, start_dt, end_dt))
        return pd.concat(dfs, ignore_index=True)

    if provider == "pjm":
        from .ingestion.grid_apis.pjm import PJMPriceProvider
        from .ingestion.grid_apis.base import ProviderConfigError
        for region in regions:
            if region not in ("us-east",):
                print(f"ERROR: PJM provider only supports us-east (got '{region}')", file=sys.stderr)
                sys.exit(1)
        start_dt = start_ts.to_pydatetime() if start_ts else None
        end_dt = end_ts.to_pydatetime() if end_ts else None
        if not start_dt or not end_dt:
            print("ERROR: --start and --end are required for live provider pjm", file=sys.stderr)
            sys.exit(1)
        p = PJMPriceProvider()
        dfs = []
        for region in regions:
            try:
                dfs.append(p.fetch_prices(region, start_dt, end_dt))
            except ProviderConfigError as e:
                print(f"ERROR: {e}", file=sys.stderr)
                sys.exit(1)
        return pd.concat(dfs, ignore_index=True)

    if provider == "entsoe":
        from .ingestion.grid_apis.entsoe import ENTSOEPriceProvider
        from .ingestion.grid_apis.base import ProviderConfigError
        start_dt = start_ts.to_pydatetime() if start_ts else None
        end_dt = end_ts.to_pydatetime() if end_ts else None
        if not start_dt or not end_dt:
            print("ERROR: --start and --end are required for live provider entsoe", file=sys.stderr)
            sys.exit(1)
        p = ENTSOEPriceProvider()
        dfs = []
        for region in regions:
            try:
                dfs.append(p.fetch_prices(region, start_dt, end_dt))
            except ProviderConfigError as e:
                print(f"ERROR: {e}", file=sys.stderr)
                sys.exit(1)
        return pd.concat(dfs, ignore_index=True)

    if provider == "eia":
        print(
            "ERROR: EIA provider is not supported for wholesale electricity prices. "
            "EIA API v2 provides demand (MWh), not prices ($/MWh). "
            "Use --price-provider=caiso (us-west) or --price-provider=pjm (us-east).",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"ERROR: Unknown price provider '{provider}'", file=sys.stderr)
    sys.exit(1)


def _load_carbon_df(args, regions):
    """Resolve --carbon-provider / --carbon-file into a canonical carbon DataFrame."""
    import pandas as pd
    from .ingestion.grid_apis.base import empty_carbon_df

    provider = args.carbon_provider

    if provider == "none":
        return empty_carbon_df()

    if provider == "csv":
        if not args.carbon_file:
            print("ERROR: --carbon-file is required when --carbon-provider=csv", file=sys.stderr)
            sys.exit(1)
        from .ingestion.grid_apis.csv_importer import CSVCarbonImporter
        importer = CSVCarbonImporter(args.carbon_file)
        df = importer.load_all()
        if not df.empty:
            df = df[df["region"].isin(regions)]
        return df

    if provider == "electricitymaps":
        from .ingestion.grid_apis.electricitymaps import ElectricityMapsCarbonProvider
        from .ingestion.grid_apis.base import ProviderConfigError
        start_ts = pd.Timestamp(args.start, tz="UTC") if args.start else None
        end_ts = pd.Timestamp(args.end, tz="UTC") if args.end else None
        if not start_ts or not end_ts:
            print("ERROR: --start and --end required for live carbon provider", file=sys.stderr)
            sys.exit(1)
        p = ElectricityMapsCarbonProvider()
        dfs = []
        for region in regions:
            try:
                dfs.append(p.fetch_carbon(region, start_ts.to_pydatetime(), end_ts.to_pydatetime()))
            except ProviderConfigError as e:
                print(f"ERROR: {e}", file=sys.stderr)
                sys.exit(1)
        return pd.concat(dfs, ignore_index=True)

    if provider == "watttime":
        from .ingestion.grid_apis.watttime import WattTimeCarbonProvider
        from .ingestion.grid_apis.base import ProviderConfigError
        start_ts = pd.Timestamp(args.start, tz="UTC") if args.start else None
        end_ts = pd.Timestamp(args.end, tz="UTC") if args.end else None
        if not start_ts or not end_ts:
            print("ERROR: --start and --end required for live carbon provider", file=sys.stderr)
            sys.exit(1)
        p = WattTimeCarbonProvider()
        dfs = []
        for region in regions:
            try:
                dfs.append(p.fetch_carbon(region, start_ts.to_pydatetime(), end_ts.to_pydatetime()))
            except ProviderConfigError as e:
                print(f"ERROR: {e}", file=sys.stderr)
                sys.exit(1)
        return pd.concat(dfs, ignore_index=True)

    print(f"ERROR: Unknown carbon provider '{provider}'", file=sys.stderr)
    sys.exit(1)


def cmd_backtest(args):
    """Run walk-forward backtest with real or CSV price/carbon data."""
    import pandas as pd
    from .backtesting.engine import BacktestEngine
    from .ingestion.job_logs import JobLogIngester
    from .models import OptimizationConfig

    regions = [r.strip() for r in args.regions.split(",")]
    start_ts = pd.Timestamp(args.start, tz="UTC") if args.start else None
    end_ts = pd.Timestamp(args.end, tz="UTC") if args.end else None

    price_df = _load_price_df(args, regions)
    if price_df.empty:
        print("ERROR: Price DataFrame is empty after loading. Check provider/file and region.", file=sys.stderr)
        sys.exit(1)

    carbon_df = _load_carbon_df(args, regions)

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
    print(f"Price provider: {args.price_provider}")
    print(f"Carbon provider: {args.carbon_provider}")
    print(f"Regions: {regions}")
    print()

    rounds = engine.run(jobs, price_df, carbon_df, start=start_ts, end=end_ts)

    if not rounds:
        print("No backtest folds produced. Check date range and data coverage.")
        sys.exit(1)

    print(f"{'Fold':>4}  {'Eval Start':>20}  {'Jobs':>5}  {'Optimizer $':>12}  {'FIFO $':>12}  {'Savings%':>9}  {'MissHrs':>7}")
    print("-" * 85)
    total_missing = 0
    for r in rounds:
        opt_cost = r.optimizer_metrics.total_energy_cost_usd if r.optimizer_metrics else 0
        missing = r.optimizer_metrics.missing_price_hours if r.optimizer_metrics else 0
        total_missing += missing
        fifo_cost = r.baseline_metrics.get("fifo", None)
        fifo_val = fifo_cost.total_energy_cost_usd if fifo_cost else float("nan")
        savings = ((fifo_val - opt_cost) / fifo_val * 100) if fifo_val > 0 else float("nan")
        miss_flag = f"  ⚠ {missing}" if missing > 0 else f"  {missing:>7}"
        print(
            f"{r.fold_index:>4}  {str(r.eval_start)[:19]:>20}  "
            f"{len(r.eval_jobs):>5}  ${opt_cost:>11.2f}  ${fifo_val:>11.2f}  "
            f"{savings:>8.1f}%{miss_flag}"
        )

    print(f"\nTotal folds: {len(rounds)}")
    if total_missing > 0:
        print(f"WARNING: {total_missing} evaluation hours used fallback price ($50/MWh) — real data was missing for those hours.")
        print("         Results should be interpreted with caution.")

    # Print all-baselines summary across folds
    all_baseline_names = list(rounds[0].baseline_metrics.keys()) if rounds else []
    if all_baseline_names:
        print("\n--- Savings vs all baselines (mean across folds) ---")
        opt_costs = [r.optimizer_metrics.total_energy_cost_usd for r in rounds if r.optimizer_metrics]
        mean_opt = sum(opt_costs) / len(opt_costs) if opt_costs else 0
        for name in all_baseline_names:
            bl_costs = [r.baseline_metrics[name].total_energy_cost_usd for r in rounds if name in r.baseline_metrics]
            if not bl_costs:
                continue
            mean_bl = sum(bl_costs) / len(bl_costs)
            savings_pct = (mean_bl - mean_opt) / mean_bl * 100 if mean_bl > 0 else float("nan")
            print(f"  vs {name:<28}  ${mean_bl:>10.2f}  →  {savings_pct:>6.1f}% savings")

    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump([r.to_dict() for r in rounds], f, indent=2, default=str)
        print(f"Results saved to: {output_path}")


def cmd_ingest(args):
    """Fetch recent price/carbon data and append to historical CSV files.

    Fetches `--days` days of data ending at the current UTC hour for the
    specified region. Appends new rows to the output CSVs (creates them if
    absent). Exits non-zero if no price provider is available for the region.

    Output CSVs are used by `scripts/retrain_forecasters.py` for model training.
    """
    import os
    from datetime import datetime, timedelta, timezone

    import pandas as pd

    from .ingestion.grid_apis.market_registry import get_registry_entry, UnsupportedMarketPriceError
    from .ingestion.grid_apis.base import empty_price_df, empty_carbon_df, ProviderConfigError

    region = args.region.strip()
    days = args.days

    # Default output paths inside the package data directory
    _data_root = Path(__file__).parent / "data"
    price_out = Path(args.price_output) if args.price_output else _data_root / "prices" / "historical_prices.csv"
    carbon_out = Path(args.carbon_output) if args.carbon_output else _data_root / "carbon" / "historical_carbon.csv"

    price_out.parent.mkdir(parents=True, exist_ok=True)
    carbon_out.parent.mkdir(parents=True, exist_ok=True)

    end_dt = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start_dt = end_dt - timedelta(days=days)

    logger.info(
        f"Ingesting {days} day(s) for region '{region}' "
        f"[{start_dt.date()} → {end_dt.date()}]"
    )

    # --- Price ingestion ---
    try:
        entry = get_registry_entry(region)
    except UnsupportedMarketPriceError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    price_df = empty_price_df()
    provider_key = entry.provider  # "caiso_oasis" | "pjm" | "entsoe" | ...

    try:
        if provider_key == "caiso_oasis":
            from .ingestion.grid_apis.caiso import CAISOPriceProvider
            p = CAISOPriceProvider()
            price_df = p.fetch_prices(region, start_dt, end_dt)
        elif provider_key == "pjm":
            from .ingestion.grid_apis.pjm import PJMPriceProvider
            api_key = os.environ.get("PJM_API_KEY", "")
            if not api_key:
                logger.warning("PJM_API_KEY not set; skipping price ingestion for us-east.")
            else:
                p = PJMPriceProvider(api_key=api_key)
                price_df = p.fetch_prices(region, start_dt, end_dt)
        elif provider_key in ("entsoe", "entso_e"):
            from .ingestion.grid_apis.entsoe import ENTSOEPriceProvider
            api_key = os.environ.get("ENTSOE_API_KEY", "")
            if not api_key:
                logger.warning("ENTSOE_API_KEY not set; skipping price ingestion for EU region.")
            else:
                p = ENTSOEPriceProvider(api_key=api_key)
                price_df = p.fetch_prices(region, start_dt, end_dt)
        else:
            logger.warning(f"No live price provider for key '{provider_key}'; skipping price ingestion.")
    except ProviderConfigError as exc:
        logger.warning(f"Price provider config error: {exc}; skipping.")
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Price ingestion failed: {exc}")
        sys.exit(1)

    if not price_df.empty:
        _append_csv(price_df, price_out, ["timestamp", "region", "price_per_mwh", "currency", "source"])
        logger.info(f"Appended {len(price_df)} price rows → {price_out}")
    else:
        logger.warning("No price data fetched; price CSV not updated.")

    # --- Carbon ingestion (best-effort, non-fatal) ---
    carbon_df = empty_carbon_df()
    try:
        em_key = os.environ.get("ELECTRICITYMAPS_API_KEY", "")
        wt_user = os.environ.get("WATTTIME_USERNAME", "")
        if em_key:
            from .ingestion.grid_apis.electricitymaps import ElectricityMapsCarbonProvider
            cp = ElectricityMapsCarbonProvider(api_key=em_key)
            carbon_df = cp.fetch_carbon(region, start_dt, end_dt)
        elif wt_user:
            from .ingestion.grid_apis.watttime import WattTimeCarbonProvider
            wt_pass = os.environ.get("WATTTIME_PASSWORD", "")
            cp = WattTimeCarbonProvider(username=wt_user, password=wt_pass)
            carbon_df = cp.fetch_carbon(region, start_dt, end_dt)
        else:
            logger.info(
                "No carbon API key set (ELECTRICITYMAPS_API_KEY or WATTTIME_USERNAME); "
                "skipping carbon ingestion."
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Carbon ingestion non-fatal error: {exc}")

    if not carbon_df.empty:
        _append_csv(carbon_df, carbon_out, ["timestamp", "region", "gco2_per_kwh", "source"])
        logger.info(f"Appended {len(carbon_df)} carbon rows → {carbon_out}")

    logger.info("Ingest complete.")


def _append_csv(df, path: Path, prefer_cols: list) -> None:
    """Append rows from *df* to *path* (creates with header if absent)."""
    cols = [c for c in prefer_cols if c in df.columns] + [
        c for c in df.columns if c not in prefer_cols
    ]
    write_header = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="", encoding="utf-8") as fh:
        df[cols].to_csv(fh, index=False, header=write_header)


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
        "--price-provider",
        required=True,
        choices=["caiso", "pjm", "entsoe", "csv"],
        help=(
            "Price data source: "
            "caiso=CAISO OASIS (us-west, no auth), "
            "pjm=PJM Data Miner (us-east, requires PJM_API_KEY), "
            "entsoe=ENTSO-E (eu-*, requires ENTSOE_API_KEY), "
            "csv=load from --price-file"
        ),
    )
    bt_parser.add_argument(
        "--price-file", default=None,
        help="Path to CSV price file (required when --price-provider=csv). "
             "Columns: timestamp, region, price_per_mwh",
    )
    bt_parser.add_argument(
        "--carbon-provider",
        default="none",
        choices=["electricitymaps", "watttime", "csv", "none"],
        help=(
            "Carbon intensity source (default: none): "
            "electricitymaps=ElectricityMaps API (requires ELECTRICITYMAPS_API_KEY), "
            "watttime=WattTime MOER (requires WATTTIME_USERNAME + WATTTIME_PASSWORD), "
            "csv=load from --carbon-file, "
            "none=skip carbon data"
        ),
    )
    bt_parser.add_argument(
        "--carbon-file", default=None,
        help="Path to CSV carbon file (required when --carbon-provider=csv). "
             "Columns: timestamp, region, gco2_per_kwh",
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

    # Ingest command
    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Fetch and append recent price/carbon data to historical CSV files",
    )
    ingest_parser.add_argument(
        "--region", default="us-east",
        help="Region to ingest (default: us-east)",
    )
    ingest_parser.add_argument(
        "--days", type=int, default=1,
        help="Number of days to fetch (default: 1)",
    )
    ingest_parser.add_argument(
        "--price-output", default=None,
        help="Path to append price CSV (default: aurelius/data/prices/historical_prices.csv)",
    )
    ingest_parser.add_argument(
        "--carbon-output", default=None,
        help="Path to append carbon CSV (default: aurelius/data/carbon/historical_carbon.csv)",
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
    elif args.command == "ingest":
        cmd_ingest(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
