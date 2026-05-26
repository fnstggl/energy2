"""CLI command implementations for constraint-aware orchestration (Phase 10+11+12).

Commands:
  constraint-report             — Run classifier + engine on a scenario or snapshot
  simulate-constraint-scenario  — Run scenario, show baseline vs Aurelius table
  telemetry-check               — Show which metrics are available/missing
  topology-report               — Show topology graph summary and bad placements
  validate-connectors           — Smoke-test all fake (sandbox) connectors
  benchmark-run                 — Multi-policy benchmark run on a scenario
  benchmark-compare             — Compare two benchmark JSON files for regressions
  optimizer-regression-check    — Check optimizer regression across all scenarios
  self-metrics                  — Export Aurelius internal metrics in Prometheus format

All commands are read-only and run in recommendation_only mode.
No secrets are included in any report output.
Sandbox outputs are labeled [SANDBOX] to prevent use in economic claims.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


def _normalize_constraint_name(s: str) -> str:
    """Normalize YAML expected_primary_constraint to ConstraintType.value format.

    YAML uses 'energy_bound', 'thermal_bound', 'memory_bound_indirect', etc.
    ConstraintType.value uses 'energy', 'thermal', 'memory', etc.
    """
    if s == "memory_bound_indirect":
        return "memory"
    return s.removesuffix("_bound")


# ---------------------------------------------------------------------------
# constraint-report
# ---------------------------------------------------------------------------

def cmd_constraint_report(args) -> None:
    """Run the classifier and engine on a scenario or snapshot, print report."""
    from aurelius.constraints import ConstraintAwareEngine
    from aurelius.reporting.constraint_report import (
        format_assessment_text,
        format_engine_result_json,
        format_recommendations_text,
    )

    state = _load_state(args)
    engine = ConstraintAwareEngine()
    result = engine.run(state)

    fmt = getattr(args, "format", "text")
    if fmt == "json":
        output = format_engine_result_json(result)
    else:
        output = format_assessment_text(result.assessment)
        output += "\n"
        output += format_recommendations_text(result)

    _write_or_print(output, getattr(args, "output", None))


# ---------------------------------------------------------------------------
# simulate-constraint-scenario
# ---------------------------------------------------------------------------

def cmd_simulate_constraint_scenario(args) -> None:
    """Run a named scenario and show baseline vs Aurelius comparison table."""
    from aurelius.constraints import ConstraintAwareEngine
    from aurelius.reporting.constraint_report import format_scenario_comparison_table
    from aurelius.simulation.cluster import ClusterSimulator, list_scenarios, load_scenario

    if getattr(args, "list", False):
        names = list_scenarios()
        print("Available scenarios:")
        for name in names:
            print(f"  {name}")
        return

    scenario_name = getattr(args, "scenario", None)
    if not scenario_name:
        print("ERROR: --scenario is required (or use --list to see options)", file=sys.stderr)
        sys.exit(1)

    seed = getattr(args, "seed", 42)
    steps = getattr(args, "steps", 24)

    try:
        scenario = load_scenario(scenario_name, seed_override=seed)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    sim = ClusterSimulator(scenario.config, seed=seed)
    ticks = sim.run(steps=steps)

    engine = ConstraintAwareEngine()
    engine_results = []
    for tick in ticks:
        er = engine.run(tick.cluster_state)
        engine_results.append(er)

    tick_metrics = [t.metrics for t in ticks]
    report = format_scenario_comparison_table(
        scenario_name=scenario_name,
        tick_metrics=tick_metrics,
        engine_results=engine_results,
    )

    _write_or_print(report, getattr(args, "output", None))

    if scenario.expected_primary_constraint:
        constraint_counts: dict[str, int] = {}
        for er in engine_results:
            bc = er.assessment.binding_constraint
            key = bc.value if bc else "none"
            constraint_counts[key] = constraint_counts.get(key, 0) + 1
        if constraint_counts:
            dominant = max(constraint_counts, key=constraint_counts.__getitem__)
            expected = scenario.expected_primary_constraint
            expected_norm = _normalize_constraint_name(expected)
            match_str = "MATCHES" if dominant == expected_norm else "MISMATCH"
            print(
                f"\nSCENARIO VALIDATION: expected={expected!r} "
                f"dominant={dominant!r} [{match_str}]"
            )


# ---------------------------------------------------------------------------
# telemetry-check
# ---------------------------------------------------------------------------

def cmd_telemetry_check(args) -> None:
    """Show which constraint signals are available in a scenario or snapshot."""
    from aurelius.constraints import ConstraintClassifier
    from aurelius.reporting.constraint_report import format_telemetry_check_text

    state = _load_state(args)
    classifier = ConstraintClassifier()
    assessment = classifier.assess(state)
    report = format_telemetry_check_text(assessment)
    _write_or_print(report, getattr(args, "output", None))


# ---------------------------------------------------------------------------
# topology-report
# ---------------------------------------------------------------------------

def cmd_topology_report(args) -> None:
    """Show topology graph summary and bad placements."""
    from aurelius.reporting.constraint_report import format_topology_report_text

    state = _load_state(args)
    report = format_topology_report_text(state)
    _write_or_print(report, getattr(args, "output", None))


# ---------------------------------------------------------------------------
# validate-connectors
# ---------------------------------------------------------------------------

def cmd_validate_connectors(args) -> None:
    """Smoke-test all fake (sandbox) connectors via the same code paths as real integrations."""
    from aurelius.reporting.constraint_report import format_validate_connectors_report

    results: list[dict] = [
        _validate_fake_prometheus(),
        _validate_dcgm_adapter(),
        _validate_vllm_adapter(),
        _validate_triton_adapter(),
        _validate_ray_serve_adapter(),
        _validate_kubernetes_connector(),
        _validate_topology_parser(),
        _validate_simulator_state_roundtrip(),
        _validate_classifier_on_simulator(),
        _validate_engine_pipeline(),
    ]

    report = format_validate_connectors_report(results)
    print(report)

    if not all(r.get("passed", False) for r in results):
        sys.exit(1)


# ---------------------------------------------------------------------------
# benchmark-run  (Phase 11)
# ---------------------------------------------------------------------------

def cmd_benchmark_run(args) -> None:
    """Run multi-policy constraint-aware benchmark on a scenario and save report."""
    from aurelius.benchmarks import ConstraintBenchmarkRunner
    from aurelius.simulation.cluster.scenarios import list_scenarios

    scenario_name = getattr(args, "scenario", None)
    run_all = getattr(args, "all_scenarios", False)
    seed = getattr(args, "seed", 42)
    steps = getattr(args, "steps", 24)
    output_dir = Path(getattr(args, "output_dir", "benchmarks/results"))
    output_dir.mkdir(parents=True, exist_ok=True)
    fmt = getattr(args, "format", "text")

    runner = ConstraintBenchmarkRunner()

    if run_all:
        scenarios = list_scenarios()
    elif scenario_name:
        scenarios = [scenario_name]
    else:
        print("ERROR: provide --scenario or --all-scenarios", file=sys.stderr)
        sys.exit(1)

    all_results = {}
    any_regression = False

    for sc_name in scenarios:
        print(f"\nRunning benchmark: {sc_name} (seed={seed}, steps={steps}) [SANDBOX]")
        try:
            result = runner.run_scenario(sc_name, seed=seed, steps=steps)
        except Exception as exc:
            print(f"  ERROR: {exc}", file=sys.stderr)
            continue

        all_results[sc_name] = result

        # Print report
        if fmt == "json":
            print(json.dumps(result.to_dict(), indent=2))
        else:
            print(result.report.to_text())

        # Save JSON
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        json_path = output_dir / f"constraint_bench_{sc_name}_{ts}.json"
        json_path.write_text(json.dumps(result.to_dict(), indent=2))
        print(f"\nSaved: {json_path}")

        if result.report.regression_flags:
            any_regression = True

    if any_regression:
        print("\n⚠  Regression flags detected. Check reports above.", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# benchmark-compare  (Phase 11)
# ---------------------------------------------------------------------------

def cmd_benchmark_compare(args) -> None:
    """Compare two constraint-aware benchmark JSON files for regressions."""
    from aurelius.benchmarks import BenchmarkRegressionChecker

    baseline_path = getattr(args, "baseline", None)
    current_path = getattr(args, "current", None)

    if not baseline_path or not current_path:
        print("ERROR: --baseline and --current are required", file=sys.stderr)
        sys.exit(1)

    policy = getattr(args, "policy", "constraint_aware")
    checker = BenchmarkRegressionChecker()

    try:
        result = checker.compare_files(baseline_path, current_path, policy=policy)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)

    print(result.to_text())

    if not result.passed:
        sys.exit(1)


# ---------------------------------------------------------------------------
# optimizer-regression-check  (Phase 11)
# ---------------------------------------------------------------------------

def cmd_optimizer_regression_check(args) -> None:
    """Run all scenarios and check for safety/stability regressions.

    Checks:
      - No SLA violations increased vs FIFO baseline
      - Migration churn stays bounded
      - Scorecard weighted_score above minimum threshold
      - Constraint classifier predicts expected primary constraint
    """
    from aurelius.benchmarks import ConstraintBenchmarkRunner
    from aurelius.simulation.cluster.scenarios import list_scenarios

    seed = getattr(args, "seed", 42)
    steps = getattr(args, "steps", 24)
    min_score = getattr(args, "min_score", 0.4)

    runner = ConstraintBenchmarkRunner()
    scenarios = list_scenarios()

    print(f"Optimizer regression check — {len(scenarios)} scenarios, seed={seed}, steps={steps}")
    print(f"Minimum scorecard threshold: {min_score}\n[SANDBOX]\n")

    failures: list[str] = []
    warnings: list[str] = []

    for sc_name in scenarios:
        try:
            result = runner.run_scenario(sc_name, seed=seed, steps=steps)
        except Exception as exc:
            failures.append(f"{sc_name}: ERROR — {exc}")
            continue

        report = result.report
        sc = report.scorecard

        # SLA regression check
        fifo_kpi = report.aggregated.get("fifo")
        ca_kpi = report.aggregated.get("constraint_aware")
        if fifo_kpi and ca_kpi:
            if ca_kpi.total_sla_violations > fifo_kpi.total_sla_violations:
                failures.append(
                    f"{sc_name}: SLA regression — constraint_aware has "
                    f"{ca_kpi.total_sla_violations} violations vs fifo "
                    f"{fifo_kpi.total_sla_violations}"
                )

        # Migration churn check
        if ca_kpi and ca_kpi.total_migrations > steps * 2:
            warnings.append(
                f"{sc_name}: HIGH migration churn ({ca_kpi.total_migrations} > threshold {steps * 2})"
            )

        # Scorecard threshold
        if sc.weighted_score < min_score:
            failures.append(
                f"{sc_name}: scorecard {sc.weighted_score:.3f} < minimum {min_score}"
            )

        # Constraint match
        if not report.constraint_match and report.expected_primary_constraint:
            warnings.append(
                f"{sc_name}: constraint mismatch — expected={report.expected_primary_constraint!r} "
                f"observed={report.observed_dominant_constraint!r}"
            )

        status = "PASS" if sc_name not in [f.split(":")[0] for f in failures] else "FAIL"
        print(f"  {status}  {sc_name:<50} score={sc.weighted_score:.3f}  "
              f"SLA={ca_kpi.total_sla_violations if ca_kpi else 'N/A'}  "
              f"mig={ca_kpi.total_migrations if ca_kpi else 'N/A'}  "
              f"constraint={'MATCH' if report.constraint_match else 'MISMATCH'}")

    print()
    if warnings:
        print("Warnings:")
        for w in warnings:
            print(f"  ⚠  {w}")
        print()

    if failures:
        print("Failures:")
        for f in failures:
            print(f"  ✗  {f}")
        print(f"\nResult: FAIL ({len(failures)} failure(s))")
        sys.exit(1)
    else:
        print(f"Result: PASS — all {len(scenarios)} scenarios passed regression check")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _load_state(args):
    """Load ClusterState from --config, --scenario, or --snapshot args."""
    from aurelius.simulation.cluster import ClusterSimulator, load_scenario

    scenario_name = getattr(args, "scenario", None)
    snapshot_path = getattr(args, "snapshot", None)
    config_path = getattr(args, "config", None)
    seed = getattr(args, "seed", 42)
    steps = getattr(args, "steps", 1)

    if config_path:
        return _assemble_state_from_config(config_path)

    if scenario_name:
        scenario = load_scenario(scenario_name, seed_override=seed)
        sim = ClusterSimulator(scenario.config, seed=seed)
        warm_steps = max(0, steps - 1)
        if warm_steps > 0:
            sim.run(steps=warm_steps)
        return sim.get_cluster_state()

    if snapshot_path:
        path = Path(snapshot_path)
        if not path.exists():
            print(f"ERROR: snapshot file not found: {path}", file=sys.stderr)
            sys.exit(1)
        try:
            raw = json.loads(path.read_text())
            from aurelius.state.models import ClusterState
            return ClusterState.from_dict(raw)
        except Exception as exc:
            print(f"ERROR: failed to load snapshot: {exc}", file=sys.stderr)
            sys.exit(1)

    print("ERROR: provide --scenario or --snapshot", file=sys.stderr)
    sys.exit(1)


def _assemble_state_from_config(config_path):
    """Assemble a ClusterState from a connector-fixture config via real adapters.

    Proves the audit's missing path: connector fixtures → production adapters →
    build_cluster_state → ClusterState (consumed by classifier/engine).

    Config JSON schema (all optional except a source):
      {
        "from_scenario": "thermal_hotspot_mixed_cluster",  # source fixture text
        "steps": 8,
        "seed": 42,
        "default_region": "us-east",
        "nodes": [{"node_id": "hot-node0", "region": "us-east"}],
        "services": [{"service_id": "llm-inference", "region": "us-east"}],
        # OR point at on-disk Prometheus exposition text:
        "dcgm_text_files": [{"path": "...", "node_id": "...", "region": "..."}],
        "vllm_text_files": [{"path": "...", "service_id_prefix": "...", "region": "..."}]
      }
    """
    from aurelius.connectors.dcgm import DCGMAdapter
    from aurelius.connectors.metric_mapping import dcgm_registry, vllm_registry
    from aurelius.connectors.prometheus import FakePrometheusClient
    from aurelius.connectors.vllm import VLLMAdapter
    from aurelius.state import build_cluster_state

    path = Path(config_path)
    if not path.exists():
        print(f"ERROR: config file not found: {path}", file=sys.stderr)
        sys.exit(1)
    try:
        cfg = json.loads(path.read_text())
    except Exception as exc:
        print(f"ERROR: failed to parse config: {exc}", file=sys.stderr)
        sys.exit(1)

    now = datetime.now(tz=timezone.utc)
    default_region = cfg.get("default_region")
    gpu_states = []
    services = []
    source_metadata = {}

    # Build a per-(node/service) DCGM/vLLM text source.
    def _dcgm_from_text(text, node_id, region):
        client = FakePrometheusClient(prometheus_text=text)
        snap = client.fetch_snapshot(dcgm_registry(), source=f"dcgm:{node_id}")
        return DCGMAdapter(client).normalize_gpus(snap, node_id=node_id,
                                                  region=region, timestamp=now)

    def _vllm_from_text(text, prefix, region):
        client = FakePrometheusClient(prometheus_text=text)
        snap = client.fetch_snapshot(vllm_registry(), source=f"vllm:{prefix}")
        svcs = VLLMAdapter(client).normalize_all_services(snap,
                                                          service_id_prefix=prefix,
                                                          timestamp=now)
        # vLLM services carry no region; tag with the configured region.
        from dataclasses import replace as _replace
        return [_replace(s, region=region) for s in svcs] if region else svcs

    scenario_name = cfg.get("from_scenario")
    if scenario_name:
        from aurelius.simulation.cluster import ClusterSimulator, load_scenario
        sc = load_scenario(scenario_name, seed_override=cfg.get("seed", 42))
        sim = ClusterSimulator(sc.config, seed=cfg.get("seed", 42))
        warm = max(0, int(cfg.get("steps", 1)) - 1)
        if warm:
            sim.run_metrics_only(warm)
        for node in cfg.get("nodes", []):
            try:
                text = sim.get_dcgm_prometheus_text(node["node_id"])
                gpu_states.extend(_dcgm_from_text(text, node["node_id"],
                                                  node.get("region", default_region)))
                source_metadata[f"dcgm:{node['node_id']}"] = {"present": True}
            except Exception as exc:
                source_metadata[f"dcgm:{node['node_id']}"] = {
                    "present": False, "error": type(exc).__name__}
        for svc in cfg.get("services", []):
            try:
                text = sim.get_vllm_prometheus_text(svc["service_id"])
                services.extend(_vllm_from_text(text, svc["service_id"],
                                                svc.get("region", default_region)))
                source_metadata[f"vllm:{svc['service_id']}"] = {"present": True}
            except Exception as exc:
                source_metadata[f"vllm:{svc['service_id']}"] = {
                    "present": False, "error": type(exc).__name__}

    for entry in cfg.get("dcgm_text_files", []):
        try:
            text = Path(entry["path"]).read_text()
            gpu_states.extend(_dcgm_from_text(text, entry["node_id"],
                                              entry.get("region", default_region)))
            source_metadata[f"dcgm:{entry['node_id']}"] = {"present": True}
        except Exception as exc:
            source_metadata[f"dcgm:{entry.get('node_id', '?')}"] = {
                "present": False, "error": type(exc).__name__}

    for entry in cfg.get("vllm_text_files", []):
        try:
            text = Path(entry["path"]).read_text()
            services.extend(_vllm_from_text(text, entry.get("service_id_prefix", "vllm"),
                                            entry.get("region", default_region)))
            source_metadata[f"vllm:{entry.get('service_id_prefix', 'vllm')}"] = {"present": True}
        except Exception as exc:
            source_metadata[f"vllm:{entry.get('service_id_prefix', '?')}"] = {
                "present": False, "error": type(exc).__name__}

    if not gpu_states and not services:
        print("ERROR: config produced no telemetry (need from_scenario/nodes/services "
              "or dcgm_text_files/vllm_text_files)", file=sys.stderr)
        sys.exit(1)

    return build_cluster_state(
        timestamp=now,
        gpu_states=gpu_states or None,
        inference_services=services or None,
        source_metadata=source_metadata,
        default_region=default_region,
    )


def _write_or_print(text: str, output_path) -> None:
    if output_path:
        Path(output_path).write_text(text)
        print(f"Report written to: {output_path}")
    else:
        print(text)


# ---------------------------------------------------------------------------
# validate-connectors helpers — each returns a result dict
# ---------------------------------------------------------------------------

def _validate_fake_prometheus() -> dict:
    name = "FakePrometheusClient"
    try:
        from aurelius.connectors.prometheus import FakePrometheusClient, RawMetricResult

        fixtures = {
            "gpu_util": [{"labels": {"gpu": "0", "node": "node-0"}, "value": "75.5"}],
        }
        client = FakePrometheusClient(fixtures=fixtures)
        resp = client.query("gpu_util")
        assert isinstance(resp, RawMetricResult)
        assert len(resp.values) == 1
        assert resp.values[0].value == 75.5
        return {"name": name, "passed": True, "detail": "query returns fixture data correctly"}
    except Exception as exc:
        return {"name": name, "passed": False, "error": str(exc)}


def _validate_dcgm_adapter() -> dict:
    name = "DCGMAdapter (via simulator text)"
    try:
        from aurelius.connectors.dcgm import DCGMAdapter, dcgm_registry
        from aurelius.connectors.prometheus import (
            FakePrometheusClient,
            PrometheusTelemetryConnector,
        )
        from aurelius.simulation.cluster import ClusterSimulator, load_scenario

        scenario = load_scenario("energy_price_arbitrage_multiregion", seed_override=42)
        sim = ClusterSimulator(scenario.config, seed=42)
        ticks = sim.run(steps=2)
        tick = ticks[-1]

        node_id = list(tick.dcgm_texts.keys())[0]
        dcgm_text = tick.dcgm_texts[node_id]

        # Use the registry-based path (same as production connector path)
        client = FakePrometheusClient(prometheus_text=dcgm_text)
        reg = dcgm_registry()
        connector = PrometheusTelemetryConnector(client, reg, source=f"dcgm-{node_id}")
        snap = connector.fetch_snapshot()

        adapter = DCGMAdapter()
        gpus = adapter.normalize_gpus(snap, node_id=node_id, region="us-east")
        assert len(gpus) > 0, "no GPUs parsed"
        gpu = gpus[0]
        assert gpu.util_pct is not None, "util_pct is None"
        return {
            "name": name, "passed": True,
            "detail": f"parsed {len(gpus)} GPU(s) from simulator DCGM text"
        }
    except Exception as exc:
        return {"name": name, "passed": False, "error": str(exc)}


def _validate_vllm_adapter() -> dict:
    name = "VLLMAdapter (via simulator text)"
    try:
        from aurelius.connectors.prometheus import (
            FakePrometheusClient,
            PrometheusTelemetryConnector,
        )
        from aurelius.connectors.vllm import VLLMAdapter, vllm_registry
        from aurelius.simulation.cluster import ClusterSimulator, load_scenario

        scenario = load_scenario("queue_surge_latency_sensitive", seed_override=42)
        sim = ClusterSimulator(scenario.config, seed=42)
        ticks = sim.run(steps=3)
        tick = ticks[-1]

        if not tick.vllm_texts:
            return {
                "name": name, "passed": True,
                "detail": "no vLLM services in this scenario (acceptable)"
            }

        svc_id = list(tick.vllm_texts.keys())[0]
        vllm_text = tick.vllm_texts[svc_id]

        client = FakePrometheusClient(prometheus_text=vllm_text)
        reg = vllm_registry()
        connector = PrometheusTelemetryConnector(client, reg, source=f"vllm-{svc_id}")
        snap = connector.fetch_snapshot()

        adapter = VLLMAdapter()
        svc = adapter.normalize_inference_state(snap, service_id=svc_id)
        assert svc.service_id == svc_id
        assert svc.engine == "vllm"
        return {
            "name": name, "passed": True,
            "detail": f"parsed service {svc_id} from simulator vLLM text"
        }
    except Exception as exc:
        return {"name": name, "passed": False, "error": str(exc)}


def _validate_triton_adapter() -> dict:
    name = "TritonAdapter (fixture)"
    try:
        from aurelius.connectors.metric_mapping import triton_registry
        from aurelius.connectors.prometheus import (
            FakePrometheusClient,
            PrometheusTelemetryConnector,
        )
        from aurelius.connectors.triton import TritonAdapter

        fixture_text = (
            "# HELP nv_inference_request_success Triton successful requests\n"
            "# TYPE nv_inference_request_success counter\n"
            'nv_inference_request_success{model="bert-large",version="1"} 1234.0\n'
            "# HELP nv_inference_queue_duration_us Queue duration microseconds\n"
            "# TYPE nv_inference_queue_duration_us gauge\n"
            'nv_inference_queue_duration_us{model="bert-large"} 45000.0\n'
        )
        client = FakePrometheusClient(prometheus_text=fixture_text)
        reg = triton_registry()
        connector = PrometheusTelemetryConnector(client, reg, source="triton-svc-0")
        snap = connector.fetch_snapshot()

        adapter = TritonAdapter()
        services = adapter.normalize_all_services(snap, service_id_prefix="triton-svc-0")
        assert len(services) >= 1
        return {
            "name": name, "passed": True,
            "detail": f"parsed {len(services)} service(s) from Triton fixture"
        }
    except Exception as exc:
        return {"name": name, "passed": False, "error": str(exc)}


def _validate_ray_serve_adapter() -> dict:
    name = "RayServeAdapter (fixture)"
    try:
        from aurelius.connectors.metric_mapping import ray_serve_registry
        from aurelius.connectors.prometheus import (
            FakePrometheusClient,
            PrometheusTelemetryConnector,
        )
        from aurelius.connectors.ray_serve import RayServeAdapter

        fixture_text = (
            "# HELP ray_serve_num_running_replicas Running replicas\n"
            "# TYPE ray_serve_num_running_replicas gauge\n"
            'ray_serve_num_running_replicas{deployment="my_deployment",application="app1"} 3.0\n'
            "# HELP ray_serve_deployment_request_counter_total Request counter\n"
            "# TYPE ray_serve_deployment_request_counter_total counter\n"
            'ray_serve_deployment_request_counter_total{deployment="my_deployment",route="/predict",application="app1"} 5000.0\n'
        )
        client = FakePrometheusClient(prometheus_text=fixture_text)
        reg = ray_serve_registry()
        connector = PrometheusTelemetryConnector(client, reg, source="ray-svc-0")
        snap = connector.fetch_snapshot()

        adapter = RayServeAdapter()
        services = adapter.normalize_all_services(snap, service_id_prefix="ray-svc-0")
        assert len(services) >= 1
        return {
            "name": name, "passed": True,
            "detail": f"parsed {len(services)} service(s) from Ray Serve fixture"
        }
    except Exception as exc:
        return {"name": name, "passed": False, "error": str(exc)}


def _validate_kubernetes_connector() -> dict:
    name = "FakeKubernetesConnector"
    try:
        from aurelius.connectors.kubernetes import FakeKubernetesConnector
        from aurelius.simulation.cluster import ClusterSimulator, load_scenario

        scenario = load_scenario("energy_price_arbitrage_multiregion", seed_override=42)
        sim = ClusterSimulator(scenario.config, seed=42)
        ticks = sim.run(steps=2)
        tick = ticks[-1]

        # node_list/pod_list come from the simulator as list of dicts
        node_list = tick.k8s_node_list.get("items", []) if isinstance(tick.k8s_node_list, dict) else tick.k8s_node_list
        pod_list = tick.k8s_pod_list.get("items", []) if isinstance(tick.k8s_pod_list, dict) else tick.k8s_pod_list

        connector = FakeKubernetesConnector(
            node_list=node_list,
            pod_list=pod_list,
        )
        snapshot = connector.collect()
        assert snapshot is not None, "snapshot is None"
        assert len(snapshot.nodes) > 0, "no nodes"
        pod_count = len(snapshot.pods)
        return {
            "name": name, "passed": True,
            "detail": f"parsed {len(snapshot.nodes)} node(s), {pod_count} pod placement(s)"
        }
    except Exception as exc:
        return {"name": name, "passed": False, "error": str(exc)}


def _validate_topology_parser() -> dict:
    name = "Topology parser (nvidia-smi topo)"
    try:
        from aurelius.connectors.topology import build_topology_state, parse_nvidia_smi_topo
        from aurelius.simulation.cluster import ClusterSimulator, load_scenario
        from aurelius.state.models import Provenance

        scenario = load_scenario("topology_fragmentation_h100", seed_override=42)
        sim = ClusterSimulator(scenario.config, seed=42)
        ticks = sim.run(steps=2)
        tick = ticks[-1]

        if not tick.topology_texts:
            return {
                "name": name, "passed": True,
                "detail": "no topology texts in scenario (acceptable)"
            }

        node_id = list(tick.topology_texts.keys())[0]
        topo_text = tick.topology_texts[node_id]
        gpu_ids, pair_levels, numa_affinity = parse_nvidia_smi_topo(topo_text)
        if not gpu_ids:
            return {"name": name, "passed": True, "detail": "parse returned empty (acceptable)"}

        ts = datetime.now(tz=timezone.utc)
        prov = Provenance(source="topology-test", fetched_at=ts, confidence="medium", is_sandbox=True)
        topo_state = build_topology_state(
            node_id=node_id,
            gpu_ids=gpu_ids,
            uuid_map={},
            pair_levels=pair_levels,
            numa_affinity=numa_affinity,
            ts=ts,
            provenance=prov,
        )
        return {
            "name": name, "passed": True,
            "detail": f"parsed topology: {len(topo_state.gpu_uuids)} GPUs, {len(topo_state.pair_levels)} pairs"
        }
    except Exception as exc:
        return {"name": name, "passed": False, "error": str(exc)}


def _validate_simulator_state_roundtrip() -> dict:
    name = "ClusterSimulator → ClusterState roundtrip"
    try:
        from aurelius.simulation.cluster import ClusterSimulator, load_scenario

        scenario = load_scenario("energy_price_arbitrage_multiregion", seed_override=42)
        sim = ClusterSimulator(scenario.config, seed=42)
        sim.run(steps=3)
        state = sim.get_cluster_state()

        assert state.provenance.is_sandbox, "is_sandbox not True"
        assert state.timestamp is not None, "timestamp is None"
        assert state.timestamp.tzinfo is not None, "timestamp is not UTC-aware"

        state_dict = state.to_dict()
        assert "timestamp" in state_dict
        assert "regions" in state_dict

        node_count = sum(len(r.nodes) for r in state.regions.values())
        return {
            "name": name, "passed": True,
            "detail": (
                f"ClusterState: {len(state.regions)} regions, "
                f"{node_count} nodes, is_sandbox=True"
            )
        }
    except Exception as exc:
        return {"name": name, "passed": False, "error": str(exc)}


def _validate_classifier_on_simulator() -> dict:
    name = "ConstraintClassifier on simulator state"
    try:
        from aurelius.constraints import ConstraintClassifier
        from aurelius.simulation.cluster import ClusterSimulator, load_scenario

        scenario = load_scenario("energy_price_arbitrage_multiregion", seed_override=42)
        sim = ClusterSimulator(scenario.config, seed=42)
        sim.run(steps=5)
        state = sim.get_cluster_state()

        classifier = ConstraintClassifier()
        assessment = classifier.assess(state)

        assert assessment.confidence is not None
        assert 0.0 <= assessment.confidence <= 1.0
        assert isinstance(assessment.missing_signals, list)

        return {
            "name": name, "passed": True,
            "detail": (
                f"binding={assessment.binding_constraint.value if assessment.binding_constraint else 'none'} "
                f"confidence={assessment.confidence:.2f} "
                f"missing={len(assessment.missing_signals)}"
            )
        }
    except Exception as exc:
        return {"name": name, "passed": False, "error": str(exc)}


def _validate_engine_pipeline() -> dict:
    name = "Full engine pipeline (classifier → cost model → recommendations)"
    try:
        from aurelius.constraints import ConstraintAwareEngine
        from aurelius.simulation.cluster import ClusterSimulator, load_scenario

        scenario = load_scenario("queue_surge_latency_sensitive", seed_override=42)
        sim = ClusterSimulator(scenario.config, seed=42)
        sim.run(steps=8)
        state = sim.get_cluster_state()

        engine = ConstraintAwareEngine()
        result = engine.run(state)

        assert result.assessment is not None
        assert isinstance(result.recommendations, list)
        for rec in result.recommendations:
            assert rec.implementation_mode == "recommendation_only"
            assert rec.provenance.is_sandbox

        return {
            "name": name, "passed": True,
            "detail": (
                f"produced {len(result.recommendations)} recommendations "
                f"({result.actionable_count} actionable, {result.noop_count} KEEP) "
                f"in {result.elapsed_ms:.1f}ms"
            )
        }
    except Exception as exc:
        return {"name": name, "passed": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# self-metrics  (Phase 12)
# ---------------------------------------------------------------------------

def cmd_self_metrics(args) -> None:
    """Export Aurelius internal operational metrics in Prometheus exposition format.

    Runs the engine on a simulator scenario to populate the observer,
    then prints the Prometheus text output.

    In production, hook this to a /self-metrics HTTP endpoint (see API server).
    All outputs are [SANDBOX] labeled when driven from the simulator.
    """
    from aurelius.constraints import AureliusObserver, ConstraintAwareEngine
    from aurelius.simulation.cluster import ClusterSimulator, load_scenario

    scenario_name = getattr(args, "scenario", "energy_price_arbitrage_multiregion") or \
        "energy_price_arbitrage_multiregion"
    steps = getattr(args, "steps", 10)
    fmt = getattr(args, "format", "prometheus")

    observer = AureliusObserver()
    engine = ConstraintAwareEngine()

    try:
        scenario = load_scenario(scenario_name, seed_override=42)
    except Exception as exc:
        print(f"[ERROR] Could not load scenario {scenario_name!r}: {exc}", file=sys.stderr)
        sys.exit(1)

    sim = ClusterSimulator(scenario.config, seed=42)

    # Simulate fake connector health (all healthy in sandbox)
    observer.record_connector_health("dcgm", is_healthy=True)
    observer.record_connector_health("vllm", is_healthy=True)
    observer.record_connector_health("kubernetes", is_healthy=True)
    observer.record_connector_health("topology", is_healthy=True)

    for _ in range(steps):
        sim.tick()
        state = sim.get_cluster_state()
        result = engine.run(state)
        observer.record_engine_result(result)

    if fmt == "json":
        import json as _json
        m = observer.get_metrics()
        print(_json.dumps({
            "constraints_detected": m.constraints_detected,
            "recommendations_generated": m.recommendations_generated,
            "recommendations_blocked_by_sla": m.recommendations_blocked_by_sla,
            "estimated_net_savings_dollars": m.estimated_net_savings_dollars,
            "confidence_current": m.confidence_current,
            "connector_health": m.connector_health,
            "stale_data_count": m.stale_data_count,
            "total_engine_cycles": m.total_engine_cycles,
        }, indent=2))
    else:
        print("[SANDBOX] Aurelius internal metrics (Prometheus text format)")
        print("# Driven from simulator — for illustration only")
        print()
        print(observer.to_prometheus_text())
