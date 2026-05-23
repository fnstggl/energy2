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

    if provider == "pjm-rt":
        from .ingestion.grid_apis.base import ProviderConfigError
        from .ingestion.grid_apis.pjm import PJMRealtimePriceProvider
        for region in regions:
            if region not in ("us-east",):
                print(f"ERROR: PJM provider only supports us-east (got '{region}')", file=sys.stderr)
                sys.exit(1)
        start_dt = start_ts.to_pydatetime() if start_ts else None
        end_dt = end_ts.to_pydatetime() if end_ts else None
        if not start_dt or not end_dt:
            print("ERROR: --start and --end are required for live provider pjm-rt", file=sys.stderr)
            sys.exit(1)
        p = PJMRealtimePriceProvider()
        dfs = []
        for region in regions:
            try:
                dfs.append(p.fetch_prices(region, start_dt, end_dt))
            except ProviderConfigError as e:
                print(f"ERROR: {e}", file=sys.stderr)
                sys.exit(1)
        return pd.concat(dfs, ignore_index=True)

    if provider in ("ercot", "ercot-rt"):
        from .ingestion.grid_apis.base import ProviderConfigError
        from .ingestion.grid_apis.ercot import (
            ERCOTPriceProvider,
            ERCOTRealtimePriceProvider,
        )
        for region in regions:
            if region not in ("us-south",):
                print(f"ERROR: ERCOT provider only supports us-south (got '{region}')", file=sys.stderr)
                sys.exit(1)
        start_dt = start_ts.to_pydatetime() if start_ts else None
        end_dt = end_ts.to_pydatetime() if end_ts else None
        if not start_dt or not end_dt:
            print(f"ERROR: --start and --end are required for live provider {provider}", file=sys.stderr)
            sys.exit(1)
        p = ERCOTRealtimePriceProvider() if provider == "ercot-rt" else ERCOTPriceProvider()
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

    settle_price_df = None
    if getattr(args, "settlement_price_file", None):
        from .ingestion.grid_apis.csv_importer import CSVPriceImporter
        settle_price_df = CSVPriceImporter(args.settlement_price_file).load_all()
        settle_price_df = settle_price_df[settle_price_df["region"].isin(regions)]
        if settle_price_df.empty:
            print("ERROR: Settlement price file is empty for the requested regions.", file=sys.stderr)
            sys.exit(1)

    carbon_df = _load_carbon_df(args, regions)

    # Generate synthetic jobs if no job file provided
    if args.jobs_file:
        job_ingester = JobLogIngester()
        jobs = job_ingester.load_from_file(args.jobs_file)
    else:
        from datetime import datetime
        job_ingester = JobLogIngester()
        sim_start = start_ts.to_pydatetime() if start_ts else datetime.utcnow()
        # Jobs must span the FULL backtest window so every fold's eval window
        # gets a real sample. generate_synthetic clusters submissions in the
        # first 70% of duration_hours (job_logs.py:163), so divide by 0.7 to
        # ensure submissions reach the latest fold.
        if start_ts is not None and end_ts is not None:
            backtest_hours = int((end_ts - start_ts).total_seconds() / 3600)
        else:
            backtest_hours = (args.train_days + args.eval_days) * 24
        duration_hours = int(backtest_hours / 0.7) + 24
        if args.workload_filter and args.workload_mix != "realistic":
            print("ERROR: --workload-filter requires --workload-mix realistic",
                  file=sys.stderr)
            sys.exit(1)
        jobs = job_ingester.generate_synthetic(
            start_time=sim_start,
            duration_hours=duration_hours,
            num_jobs=args.num_jobs,
            regions=regions,
            seed=42,
            workload_mix=args.workload_mix,
            workload_filter=args.workload_filter,
        )

    config = OptimizationConfig()

    price_forecaster_cls = None
    if args.forecaster == "ml_quantile":
        # Mute the benign "X does not have valid feature names" warning that
        # lightgbm-via-sklearn emits when fit gets a DataFrame and predict
        # gets a numpy array. It does not affect predictions.
        import warnings
        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names",
            category=UserWarning,
        )
        try:
            import lightgbm  # noqa: F401  preflight so the error is clear
        except ImportError:
            print(
                "ERROR: --forecaster ml_quantile requires lightgbm. Install with:\n"
                "  pip install lightgbm scikit-learn",
                file=sys.stderr,
            )
            sys.exit(1)
        except OSError as exc:
            # lightgbm imports but its native lib failed to load. Most common
            # cause on macOS is the missing libomp.dylib system dependency.
            hint = ""
            if "libomp" in str(exc):
                hint = "\nOn macOS, install the OpenMP runtime:\n  brew install libomp"
            print(
                f"ERROR: lightgbm is installed but failed to load its native "
                f"library:\n  {exc}{hint}",
                file=sys.stderr,
            )
            sys.exit(1)
        from .forecasting.price_model import PriceQuantileForecaster
        price_forecaster_cls = PriceQuantileForecaster

    # Negative lambda disables the DA->RT risk adjustment (plan on raw DA).
    rt_risk_lambda = (
        None if (args.rt_risk_lambda is None or args.rt_risk_lambda < 0)
        else args.rt_risk_lambda
    )
    engine = BacktestEngine(
        method=args.method,
        train_days=args.train_days,
        eval_days=args.eval_days,
        config=config,
        price_forecaster_cls=price_forecaster_cls,
        rt_risk_lambda=rt_risk_lambda,
    )
    if args.forecaster == "oracle":
        engine.oracle_forecast = True
    if args.forecast_horizon_hours is not None:
        engine.forecast_horizon_hours = args.forecast_horizon_hours
        engine.replan_hours = args.replan_hours

    print(f"\nRunning backtest: {args.train_days}d train / {args.eval_days}d eval windows")
    print(f"Price provider: {args.price_provider} (planning signal)")
    if settle_price_df is not None:
        print("Settlement: realized prices from --settlement-price-file (RT-exposed customer)")
    else:
        print("Settlement: planning price (DA-hedged customer; pays what was planned)")
    print(f"Carbon provider: {args.carbon_provider}")
    print(f"Forecaster: {args.forecaster}"
          + ("  [DIAGNOSTIC: perfect-foresight leakage — not a real savings number]"
             if args.forecaster == "oracle" else ""))
    print(f"Workload mix: {args.workload_mix}"
          + (f"  filter={args.workload_filter}" if args.workload_filter else ""))
    if args.forecast_horizon_hours is not None:
        print(f"Rolling horizon: {args.forecast_horizon_hours}h actual DAM / "
              f"replan every {args.replan_hours}h (ML beyond horizon)")
    if rt_risk_lambda is None:
        print("DA->RT risk adjustment: OFF (planning on raw day-ahead)")
    elif settle_price_df is None:
        print("DA->RT risk adjustment: inactive (no --settlement-price-file; "
              "plan price == settle price)")
    else:
        print(f"DA->RT risk adjustment: ON (lambda={rt_risk_lambda}; optimizer "
              "plans on debiased RT estimate + spike penalty)")
    print(f"Regions: {regions}")
    print()

    rounds = engine.run(jobs, price_df, carbon_df, start=start_ts, end=end_ts,
                        settle_price_df=settle_price_df)

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
            mean_savings = mean_bl - mean_opt
            savings_pct = (mean_bl - mean_opt) / mean_bl * 100 if mean_bl > 0 else float("nan")
            print(
                f"  vs {name:<24}  saved ${mean_savings:>11,.2f}  ({savings_pct:>5.1f}%)"
                f"   [opt ${mean_opt:>10,.2f} vs ${mean_bl:>10,.2f}]"
            )

    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump([r.to_dict() for r in rounds], f, indent=2, default=str)
        print(f"Results saved to: {output_path}")


def cmd_shadow_run(args):
    """Run shadow mode: make optimizer decisions without executing workloads."""
    import pandas as pd
    from pathlib import Path
    from .shadow import LiveShadowRunner, DecisionRecorder
    from .models import OptimizationConfig
    from .ingestion.job_logs import JobLogIngester

    regions = [r.strip() for r in args.regions.split(",")]
    output_dir = Path(args.output_dir) if args.output_dir else Path("reports/shadow")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load DA price data
    from .ingestion.grid_apis.csv_importer import CSVPriceImporter
    price_df = CSVPriceImporter(args.price_file).load_all()
    if price_df.empty:
        print(f"ERROR: No price data loaded from {args.price_file}", file=sys.stderr)
        sys.exit(1)
    price_df = price_df[price_df["region"].isin(regions)]

    # Load carbon data (optional)
    carbon_df = None
    if args.carbon_file:
        from .ingestion.grid_apis.csv_importer import CSVCarbonImporter
        carbon_df = CSVCarbonImporter(args.carbon_file).load_all()

    # Resolve decision_time
    decision_time = None
    if args.decision_time:
        from datetime import timezone
        decision_time = datetime.fromisoformat(args.decision_time.replace("Z", "+00:00"))
        if decision_time.tzinfo is None:
            decision_time = decision_time.replace(tzinfo=timezone.utc)

    # Load or generate jobs
    if args.jobs_file:
        ingester = JobLogIngester(regions=regions)
        jobs = ingester.load_from_file(args.jobs_file)
        if not jobs:
            print(f"ERROR: No jobs loaded from {args.jobs_file}", file=sys.stderr)
            sys.exit(1)
        print(f"Loaded {len(jobs)} jobs from {args.jobs_file}")
    else:
        # Synthetic jobs around the decision window
        from .ingestion.job_logs import JobLogIngester as _JLI
        ingester = _JLI()
        # Use last available price timestamp as submit anchor
        price_df["timestamp"] = pd.to_datetime(price_df["timestamp"], utc=True)
        anchor = price_df["timestamp"].max().to_pydatetime()
        jobs = ingester.generate_synthetic(
            num_jobs=args.num_jobs,
            start_time=anchor - pd.Timedelta(hours=24),
            duration_hours=14 * 24,
            regions=regions,
            workload_mix="realistic",
            seed=42,
        )
        print(f"Generated {len(jobs)} synthetic jobs")

    # Build forecaster
    price_forecaster_cls = None
    price_forecaster_config = None
    if args.forecaster == "ml_quantile":
        from .forecasting.price_model import PriceQuantileForecaster, PriceModelConfig
        price_forecaster_cls = PriceQuantileForecaster
        price_forecaster_config = PriceModelConfig(seed=42, n_estimators=200, num_leaves=63)

    config = OptimizationConfig()
    runner = LiveShadowRunner(
        regions=regions,
        method="greedy",
        train_days=args.train_days,
        horizon_hours=args.horizon_hours,
        config=config,
        price_forecaster_cls=price_forecaster_cls,
        price_forecaster_config=price_forecaster_config,
    )

    records = runner.run(
        price_df=price_df,
        jobs=jobs,
        carbon_df=carbon_df,
        decision_time=decision_time,
    )

    if not records:
        print("No decisions produced. Check price data range and job submit times.")
        sys.exit(1)

    from datetime import timezone as _tz
    ts = datetime.now(_tz.utc).strftime("%Y%m%dT%H%M%SZ")
    decisions_path = output_dir / f"decisions_{ts}.jsonl"
    recorder = DecisionRecorder(output_path=decisions_path)
    recorder.save(records)

    # Print summary
    pred_savings = sum(r.predicted_savings_pct for r in records) / len(records)
    print(f"\nSHADOW RUN COMPLETE")
    print(f"  Run ID:              {runner.run_id}")
    print(f"  Jobs decided:        {len(records)}")
    print(f"  Mean predicted saving (vs CPO): {pred_savings:.1f}%")
    print(f"  Decisions saved to:  {decisions_path}")
    print(f"\nNext steps:")
    print(f"  1. Wait until scheduled jobs have run (7-14 days for RT prices to settle).")
    print(f"  2. Run: python -m aurelius.cli shadow realize \\")
    print(f"       --decisions-file {decisions_path} \\")
    print(f"       --rt-price-file <rt_settlement.csv>")
    print(f"  3. Run: python -m aurelius.cli shadow report \\")
    print(f"       --decisions-file <realized.jsonl>")


def cmd_shadow_realize(args):
    """Fill in realized RT prices for a pending shadow decisions file."""
    from pathlib import Path
    from .shadow import DecisionRecorder, RealizedSavingsCalculator
    from .ingestion.grid_apis.csv_importer import CSVPriceImporter

    decisions_path = Path(args.decisions_file)
    recorder = DecisionRecorder()
    records = recorder.load(decisions_path)
    if not records:
        print(f"ERROR: No records loaded from {decisions_path}", file=sys.stderr)
        sys.exit(1)

    # Load RT settlement prices
    rt_df = CSVPriceImporter(args.rt_price_file).load_all()
    if rt_df.empty:
        print(f"ERROR: No RT price data from {args.rt_price_file}", file=sys.stderr)
        sys.exit(1)

    calculator = RealizedSavingsCalculator(rt_df)
    realized_records = calculator.realize(records)

    n_realized = sum(1 for r in realized_records if r.is_realized)
    n_pending = len(realized_records) - n_realized

    if args.output_file:
        out_path = Path(args.output_file)
    else:
        stem = decisions_path.stem.replace("decisions_", "realized_")
        out_path = decisions_path.parent / f"{stem}.jsonl"

    recorder.save_updated(realized_records, path=out_path)
    print(f"\nREALIZATION COMPLETE")
    print(f"  Records processed:  {len(realized_records)}")
    print(f"  Records realized:   {n_realized}")
    print(f"  Records pending:    {n_pending} (RT data not available)")
    print(f"  Output saved to:    {out_path}")
    if n_realized > 0:
        realized = [r for r in realized_records if r.is_realized]
        mean_real = sum(r.realized_savings_pct for r in realized) / len(realized)
        mean_pred = sum(r.predicted_savings_pct for r in realized) / len(realized)
        print(f"\n  Mean predicted savings:  {mean_pred:.1f}%")
        print(f"  Mean realized savings:   {mean_real:.1f}%")
        delta = mean_real - mean_pred
        sign = "+" if delta >= 0 else ""
        print(f"  Delta (realized-pred):   {sign}{delta:.1f}pp")


def cmd_shadow_report(args):
    """Generate a shadow mode comparison report."""
    from pathlib import Path
    from .shadow import DecisionRecorder, ShadowReport

    decisions_path = Path(args.decisions_file)
    recorder = DecisionRecorder()
    records = recorder.load(decisions_path)
    if not records:
        print(f"ERROR: No records loaded from {decisions_path}", file=sys.stderr)
        sys.exit(1)

    report = ShadowReport.from_records(records, data_source_note=str(decisions_path))

    output_dir = Path(args.output_dir) if args.output_dir else decisions_path.parent
    paths = report.save(output_dir)

    print(report.to_text())
    print(f"\nReport saved to:")
    print(f"  JSON: {paths['json']}")
    print(f"  TXT:  {paths['txt']}")


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
        choices=["caiso", "pjm", "pjm-rt", "ercot", "ercot-rt", "entsoe", "csv"],
        help=(
            "Price data source: "
            "caiso=CAISO OASIS day-ahead (us-west, no auth), "
            "pjm=PJM Data Miner day-ahead (us-east, requires PJM_API_KEY), "
            "pjm-rt=PJM Data Miner real-time 5-min (us-east, requires PJM_API_KEY), "
            "ercot=ERCOT day-ahead SPP (us-south, requires ERCOT creds), "
            "ercot-rt=ERCOT real-time 15-min SPP (us-south, requires ERCOT creds), "
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
        "--settlement-price-file", default=None,
        help=(
            "Optional CSV of SETTLEMENT prices (what the customer actually pays, "
            "e.g. realized real-time LMP). When given, schedules are SCORED against "
            "these prices while the optimizer still plans against --price-provider "
            "data (e.g. day-ahead). This is the DA-plan / RT-settle model for an "
            "RT-exposed customer. Omit it to model a DA-hedged customer (settlement "
            "== planning price). Columns: timestamp, region, price_per_mwh"
        ),
    )
    bt_parser.add_argument(
        "--rt-risk-lambda", type=float, default=1.0,
        help=(
            "DA->RT spread risk-aversion weight (default 1.0; active only with "
            "--settlement-price-file). The optimizer plans against a risk-adjusted "
            "RT estimate: DA + learned per-(region,hour) median spread + lambda * "
            "upside spike risk (max of hour-of-day and DA-price-level signals), "
            "fit on the training window only. 0 = debias toward expected RT only; "
            "higher avoids spike-prone hours/regions more aggressively. Set a "
            "negative value to disable and plan on raw day-ahead."
        ),
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
        help=(
            "Path to a workload trace file. Accepts: "
            "(1) customer CSV with columns job_id,workload_type,submit_time,"
            "duration_hours (plus optional gpu_count,deadline,allowed_regions,...); "
            "(2) legacy internal CSV (job_id,submit_time,runtime_hours,...); "
            "(3) JSON job log. Format is auto-detected. "
            "Generates synthetic jobs if omitted."
        ),
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
        choices=[
            "greedy", "local_search",
            "greedy_migrate", "local_search_migrate",
            "greedy_migrate_dp", "local_search_migrate_dp",
        ],
        help=(
            "Optimizer method (default: greedy). The _migrate variants "
            "post-process the base schedule by trying a single mid-job "
            "region migration per job whose workload allows it "
            "(realtime_inference cannot; training/fine-tuning/batch can). "
            "The _migrate_dp variants do exact multi-migration optimization "
            "via DP over (useful_hours_done, region, num_migrations) — "
            "captures cycle-chasing on long jobs that single migration "
            "cannot. Migration cost (~6-30 min depending on workload) is "
            "modeled explicitly in the scoring."
        ),
    )
    bt_parser.add_argument(
        "--forecaster", default="seasonal_naive",
        choices=["seasonal_naive", "ml_quantile", "oracle"],
        help=(
            "Price forecaster for each fold (default: seasonal_naive). "
            "ml_quantile fits a LightGBM quantile model per fold on the "
            "training window — requires `pip install lightgbm scikit-learn`. "
            "oracle is a DIAGNOSTIC ONLY: it feeds the optimizer the actual "
            "eval-window prices (perfect foresight / intentional leakage). "
            "Use it to measure the savings ceiling with a perfect forecaster — "
            "if oracle savings >> ml_quantile savings, forecasting is the "
            "bottleneck; if they're similar, the price spread is the bottleneck. "
            "NEVER report oracle numbers as real savings."
        ),
    )
    bt_parser.add_argument(
        "--forecast-horizon-hours", type=int, default=None,
        help=(
            "Enable rolling-horizon (receding-horizon / MPC) optimization. "
            "Models production reality: day-ahead prices are published ~1 day "
            "ahead, so the scheduler re-plans daily and KNOWS the actual prices "
            "for the next N hours (published DAM), using the ML/naive forecast "
            "only beyond that. Typical value: 24 (single-day DAM) or 36. "
            "When omitted, the optimizer does a single one-shot plan over the "
            "whole eval window using the forecast for every hour (the "
            "pessimistic / pure-forecast baseline). NOT leakage — the next-day "
            "actual prices are genuinely published in production."
        ),
    )
    bt_parser.add_argument(
        "--replan-hours", type=int, default=24,
        help="Re-planning cadence for rolling-horizon mode (default: 24h = daily). "
             "Only used when --forecast-horizon-hours is set.",
    )
    bt_parser.add_argument(
        "--workload-filter", default=None,
        choices=[
            "realtime_inference", "llm_batch_inference", "fine_tuning",
            "training", "data_processing", "scheduled_batch",
            "background_maintenance",
        ],
        help=(
            "Restrict synthetic jobs to a single workload type (only valid "
            "with --workload-mix realistic). Use to measure per-workload "
            "savings separately — e.g. 'how much do we save on training jobs "
            "vs batch inference?' Without this, all 7 types are mixed and the "
            "blended number is dominated by the cost-heaviest type (training)."
        ),
    )
    bt_parser.add_argument(
        "--workload-mix", default="legacy",
        choices=["legacy", "realistic"],
        help=(
            "Synthetic job mix (default: legacy). "
            "legacy = size-based profiles (small/medium/large/xlarge), "
            "global slack range, 70%% multi-region. "
            "realistic = 7 workload-type profiles (realtime_inference, "
            "llm_batch_inference, fine_tuning, training, data_processing, "
            "scheduled_batch, background_maintenance) with per-profile slack "
            "and multi-region flexibility — gives the optimizer realistic "
            "headroom to capture price spreads."
        ),
    )
    bt_parser.add_argument(
        "--output", default=None,
        help="Save results as JSON to this path",
    )

    # --- Shadow mode subcommand ---
    shadow_parser = subparsers.add_parser(
        "shadow",
        help="Production shadow mode — make optimizer decisions without executing workloads",
    )
    shadow_subparsers = shadow_parser.add_subparsers(dest="shadow_command")

    # shadow run
    sr_parser = shadow_subparsers.add_parser(
        "run",
        help=(
            "Run shadow mode: train on historical DA prices, forecast next window, "
            "schedule submitted jobs, save decisions (no workloads are executed)."
        ),
    )
    sr_parser.add_argument(
        "--price-file", required=True,
        help="Path to DA price CSV (timestamp, region, price_per_mwh)",
    )
    sr_parser.add_argument(
        "--regions", default="us-west,us-east,us-south",
        help="Comma-separated list of regions (default: us-west,us-east,us-south)",
    )
    sr_parser.add_argument(
        "--jobs-file", default=None,
        help="Customer workload trace CSV. If absent, synthetic jobs are generated.",
    )
    sr_parser.add_argument(
        "--num-jobs", type=int, default=50,
        help="Number of synthetic jobs when --jobs-file is absent (default: 50)",
    )
    sr_parser.add_argument(
        "--carbon-file", default=None,
        help="Optional carbon intensity CSV (timestamp, region, gco2_per_kwh)",
    )
    sr_parser.add_argument(
        "--train-days", type=int, default=30,
        help="Days of history to use for forecaster training (default: 30)",
    )
    sr_parser.add_argument(
        "--horizon-hours", type=int, default=168,
        help="How far ahead to forecast and schedule in hours (default: 168)",
    )
    sr_parser.add_argument(
        "--forecaster", default="ml_quantile",
        choices=["ml_quantile", "seasonal_naive"],
        help="Forecasting method (default: ml_quantile)",
    )
    sr_parser.add_argument(
        "--decision-time", default=None,
        help=(
            "ISO 8601 UTC timestamp for 'now' (default: last available price + 1h). "
            "Example: 2026-03-01T00:00:00Z"
        ),
    )
    sr_parser.add_argument(
        "--output-dir", default=None,
        help="Directory for decisions JSONL output (default: reports/shadow/)",
    )

    # shadow realize
    srz_parser = shadow_subparsers.add_parser(
        "realize",
        help=(
            "Fill in realized RT prices for pending shadow decisions. "
            "Run after the scheduled job windows have passed."
        ),
    )
    srz_parser.add_argument(
        "--decisions-file", required=True,
        help="Path to decisions JSONL file (output of 'shadow run')",
    )
    srz_parser.add_argument(
        "--rt-price-file", required=True,
        help="Path to RT settlement price CSV (timestamp, region, price_per_mwh)",
    )
    srz_parser.add_argument(
        "--output-file", default=None,
        help="Output JSONL path (default: realized_<timestamp>.jsonl in same dir)",
    )

    # shadow report
    srp_parser = shadow_subparsers.add_parser(
        "report",
        help=(
            "Generate a shadow mode comparison report (predicted vs realized savings). "
            "Works with decisions files that have full or partial realization."
        ),
    )
    srp_parser.add_argument(
        "--decisions-file", required=True,
        help="Path to JSONL file (from 'shadow run' or 'shadow realize')",
    )
    srp_parser.add_argument(
        "--output-dir", default=None,
        help="Directory for report output (default: same dir as decisions file)",
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
    elif args.command == "shadow":
        sc = getattr(args, "shadow_command", None)
        if sc == "run":
            cmd_shadow_run(args)
        elif sc == "realize":
            cmd_shadow_realize(args)
        elif sc == "report":
            cmd_shadow_report(args)
        else:
            shadow_parser.print_help()
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
