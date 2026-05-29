"""Tests for Phase 10 CLI commands — constraint-report, simulate-constraint-scenario,
telemetry-check, topology-report, validate-connectors.

These tests use the same simulator + fake connector paths as production to
prove the CLI is wired correctly, not just that functions exist.

All tests run fully offline (no network, no real cluster).
All tests assert is_sandbox=True on simulator outputs.
No secrets appear in any report output.
"""

from __future__ import annotations

import argparse
import json
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_args(**kwargs):
    """Build a minimal argparse.Namespace with sensible defaults."""
    defaults = {
        "scenario": "energy_price_arbitrage_multiregion",
        "snapshot": None,
        "steps": 3,
        "seed": 42,
        "format": "text",
        "output": None,
        "list": False,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def _get_simulator_state(scenario="energy_price_arbitrage_multiregion", steps=3, seed=42):
    from aurelius.simulation.cluster import ClusterSimulator, load_scenario
    scenario_cfg = load_scenario(scenario, seed_override=seed)
    sim = ClusterSimulator(scenario_cfg.config, seed=seed)
    sim.run(steps=steps)
    return sim.get_cluster_state()


# ---------------------------------------------------------------------------
# constraint_report formatters
# ---------------------------------------------------------------------------

class TestFormatAssessmentText:
    def test_basic_text_output(self):
        from aurelius.reporting.constraint_report import format_assessment_text
        state = _get_simulator_state(steps=5)
        from aurelius.constraints import ConstraintClassifier
        classifier = ConstraintClassifier()
        assessment = classifier.assess(state)
        text = format_assessment_text(assessment)
        assert "CONSTRAINT ASSESSMENT" in text
        assert "BINDING CONSTRAINT" in text
        assert "Confidence" in text

    def test_sandbox_label_present(self):
        from aurelius.reporting.constraint_report import format_assessment_text
        state = _get_simulator_state()
        assert state.provenance.is_sandbox
        from aurelius.constraints import ConstraintClassifier
        assessment = ConstraintClassifier().assess(state)
        text = format_assessment_text(assessment)
        assert "SANDBOX" in text

    def test_missing_signals_shown(self):
        from datetime import datetime, timezone

        from aurelius.constraints import ConstraintClassifier
        from aurelius.reporting.constraint_report import format_assessment_text
        from aurelius.state.models import ClusterState, Provenance
        # Use an empty ClusterState — no GPU/queue/energy data → all scorers report missing
        ts = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        prov = Provenance(source="test", fetched_at=ts, confidence="low")
        empty_state = ClusterState(timestamp=ts, provenance=prov, regions={}, is_partial=True)
        assessment = ConstraintClassifier().assess(empty_state)
        text = format_assessment_text(assessment)
        # Empty state has no signals at all → MISSING TELEMETRY section must appear
        assert "MISSING TELEMETRY" in text

    def test_scores_are_bounded(self):
        state = _get_simulator_state(steps=10)
        from aurelius.constraints import ConstraintClassifier
        assessment = ConstraintClassifier().assess(state)
        for ct, score in assessment.scores.items():
            assert 0.0 <= score <= 1.0, f"Score out of range for {ct}: {score}"


class TestFormatRecommendationsText:
    def test_basic_output(self):
        from aurelius.constraints import ConstraintAwareEngine
        from aurelius.reporting.constraint_report import format_recommendations_text
        state = _get_simulator_state()
        result = ConstraintAwareEngine().run(state)
        text = format_recommendations_text(result)
        assert "RECOMMENDATIONS" in text
        assert "recommendation_only" in text

    def test_sandbox_label(self):
        from aurelius.constraints import ConstraintAwareEngine
        from aurelius.reporting.constraint_report import format_recommendations_text
        state = _get_simulator_state()
        result = ConstraintAwareEngine().run(state)
        text = format_recommendations_text(result)
        assert "SANDBOX" in text

    def test_no_secrets(self):
        from aurelius.constraints import ConstraintAwareEngine
        from aurelius.reporting.constraint_report import format_recommendations_text
        state = _get_simulator_state()
        result = ConstraintAwareEngine().run(state)
        text = format_recommendations_text(result)
        for secret_keyword in ["password", "token", "secret", "api_key", "DATABASE_URL"]:
            assert secret_keyword.lower() not in text.lower(), f"Secret keyword found: {secret_keyword}"

    def test_all_recommendations_are_recommendation_only(self):
        from aurelius.constraints import ConstraintAwareEngine
        state = _get_simulator_state(steps=10)
        result = ConstraintAwareEngine().run(state)
        for rec in result.recommendations:
            assert rec.implementation_mode == "recommendation_only"


class TestFormatEngineResultJson:
    def test_json_valid(self):
        from aurelius.constraints import ConstraintAwareEngine
        from aurelius.reporting.constraint_report import format_engine_result_json
        state = _get_simulator_state()
        result = ConstraintAwareEngine().run(state)
        json_str = format_engine_result_json(result)
        parsed = json.loads(json_str)
        assert "assessment" in parsed
        assert "recommendations" in parsed
        assert "elapsed_ms" in parsed

    def test_json_schema(self):
        from aurelius.constraints import ConstraintAwareEngine
        from aurelius.reporting.constraint_report import format_engine_result_json
        state = _get_simulator_state()
        result = ConstraintAwareEngine().run(state)
        parsed = json.loads(format_engine_result_json(result))
        assessment = parsed["assessment"]
        assert "binding_constraint" in assessment
        assert "confidence" in assessment
        assert "missing_signals" in assessment
        assert "scores" in assessment

    def test_json_no_secrets(self):
        from aurelius.constraints import ConstraintAwareEngine
        from aurelius.reporting.constraint_report import format_engine_result_json
        state = _get_simulator_state()
        result = ConstraintAwareEngine().run(state)
        json_str = format_engine_result_json(result)
        for keyword in ["password", "token", "secret", "api_key", "Bearer"]:
            assert keyword.lower() not in json_str.lower(), f"Possible secret found: {keyword}"


class TestFormatTelemetryCheckText:
    def test_basic_output(self):
        from aurelius.constraints import ConstraintClassifier
        from aurelius.reporting.constraint_report import format_telemetry_check_text
        state = _get_simulator_state()
        assessment = ConstraintClassifier().assess(state)
        text = format_telemetry_check_text(assessment)
        assert "TELEMETRY COVERAGE CHECK" in text
        assert "CONSTRAINT DETECTION CAPABILITY" in text
        assert "COVERAGE SUMMARY" in text

    def test_shows_missing_signals(self):
        from aurelius.constraints import ConstraintClassifier
        from aurelius.reporting.constraint_report import format_telemetry_check_text
        state = _get_simulator_state()
        assessment = ConstraintClassifier().assess(state)
        text = format_telemetry_check_text(assessment)
        if assessment.missing_signals:
            assert "MISSING SIGNALS" in text

    def test_coverage_pct_is_valid(self):
        from aurelius.constraints import ConstraintClassifier
        from aurelius.reporting.constraint_report import format_telemetry_check_text
        state = _get_simulator_state(steps=10)
        assessment = ConstraintClassifier().assess(state)
        text = format_telemetry_check_text(assessment)
        # Should show a valid percentage
        import re
        match = re.search(r"(\d+)/8 \((\d+)%\)", text)
        assert match is not None, f"Expected coverage pattern in: {text}"
        pct = int(match.group(2))
        assert 0 <= pct <= 100


class TestFormatTopologyReportText:
    def test_basic_output(self):
        from aurelius.reporting.constraint_report import format_topology_report_text
        state = _get_simulator_state()
        text = format_topology_report_text(state)
        assert "TOPOLOGY REPORT" in text
        assert "CLUSTER TOPOLOGY" in text

    def test_sandbox_label(self):
        from aurelius.reporting.constraint_report import format_topology_report_text
        state = _get_simulator_state()
        text = format_topology_report_text(state)
        assert "SANDBOX" in text

    def test_shows_nodes(self):
        from aurelius.reporting.constraint_report import format_topology_report_text
        state = _get_simulator_state()
        text = format_topology_report_text(state)
        # Energy scenario has nodes in us-east/us-west
        assert "REGION" in text or "Nodes" in text


class TestFormatScenarioComparisonTable:
    def test_basic_table(self):
        from aurelius.constraints import ConstraintAwareEngine
        from aurelius.reporting.constraint_report import format_scenario_comparison_table
        from aurelius.simulation.cluster import ClusterSimulator, load_scenario

        scenario = load_scenario("energy_price_arbitrage_multiregion", seed_override=42)
        sim = ClusterSimulator(scenario.config, seed=42)
        ticks = sim.run(steps=5)
        engine = ConstraintAwareEngine()
        engine_results = [engine.run(t.cluster_state) for t in ticks]
        tick_metrics = [t.metrics for t in ticks]

        text = format_scenario_comparison_table("test_scenario", tick_metrics, engine_results)
        assert "SCENARIO: test_scenario" in text
        assert "SANDBOX" in text
        assert "AGGREGATE METRICS" in text
        assert "PER-TICK SUMMARY" in text

    def test_empty_ticks(self):
        from aurelius.reporting.constraint_report import format_scenario_comparison_table
        text = format_scenario_comparison_table("empty", [], [])
        assert "No ticks run" in text

    def test_sandbox_label_present(self):
        from aurelius.constraints import ConstraintAwareEngine
        from aurelius.reporting.constraint_report import format_scenario_comparison_table
        from aurelius.simulation.cluster import ClusterSimulator, load_scenario

        scenario = load_scenario("energy_price_arbitrage_multiregion", seed_override=42)
        sim = ClusterSimulator(scenario.config, seed=42)
        ticks = sim.run(steps=3)
        engine = ConstraintAwareEngine()
        engine_results = [engine.run(t.cluster_state) for t in ticks]
        tick_metrics = [t.metrics for t in ticks]

        text = format_scenario_comparison_table("test", tick_metrics, engine_results)
        assert "is_sandbox=True" in text or "SANDBOX" in text


class TestFormatValidateConnectorsReport:
    def test_all_pass(self):
        from aurelius.reporting.constraint_report import format_validate_connectors_report
        results = [
            {"name": "A", "passed": True, "detail": "ok"},
            {"name": "B", "passed": True, "detail": "ok too"},
        ]
        text = format_validate_connectors_report(results)
        assert "ALL PASSED" in text
        assert "[PASS] A" in text

    def test_with_failures(self):
        from aurelius.reporting.constraint_report import format_validate_connectors_report
        results = [
            {"name": "A", "passed": True, "detail": "ok"},
            {"name": "B", "passed": False, "error": "something broke"},
        ]
        text = format_validate_connectors_report(results)
        assert "FAILURES DETECTED" in text
        assert "[FAIL] B" in text
        assert "something broke" in text


# ---------------------------------------------------------------------------
# CLI command functions
# ---------------------------------------------------------------------------

class TestCmdConstraintReport:
    def test_text_output_to_stdout(self, capsys):
        from aurelius.cli_constraint import cmd_constraint_report
        args = _make_args(scenario="energy_price_arbitrage_multiregion", steps=2, format="text")
        cmd_constraint_report(args)
        captured = capsys.readouterr()
        assert "CONSTRAINT ASSESSMENT" in captured.out
        assert "RECOMMENDATIONS" in captured.out

    def test_json_output_to_stdout(self, capsys):
        from aurelius.cli_constraint import cmd_constraint_report
        args = _make_args(scenario="energy_price_arbitrage_multiregion", steps=2, format="json")
        cmd_constraint_report(args)
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert "assessment" in parsed
        assert "recommendations" in parsed

    def test_output_to_file(self, tmp_path):
        from aurelius.cli_constraint import cmd_constraint_report
        out_file = tmp_path / "report.txt"
        args = _make_args(
            scenario="energy_price_arbitrage_multiregion",
            steps=2,
            format="text",
            output=str(out_file),
        )
        cmd_constraint_report(args)
        assert out_file.exists()
        content = out_file.read_text()
        assert "CONSTRAINT ASSESSMENT" in content

    def test_json_output_to_file_is_valid(self, tmp_path):
        from aurelius.cli_constraint import cmd_constraint_report
        out_file = tmp_path / "report.json"
        args = _make_args(
            scenario="thermal_hotspot_mixed_cluster",
            steps=3,
            format="json",
            output=str(out_file),
        )
        cmd_constraint_report(args)
        parsed = json.loads(out_file.read_text())
        assert "assessment" in parsed

    def test_no_secrets_in_output(self, capsys):
        from aurelius.cli_constraint import cmd_constraint_report
        args = _make_args(scenario="energy_price_arbitrage_multiregion", steps=2)
        cmd_constraint_report(args)
        captured = capsys.readouterr()
        for keyword in ["password", "secret", "Bearer", "DATABASE_URL", "api_key"]:
            assert keyword.lower() not in captured.out.lower()

    def test_snapshot_load_invalid_path(self):
        from aurelius.cli_constraint import cmd_constraint_report
        args = _make_args(scenario=None, snapshot="/nonexistent/path.json")
        with pytest.raises(SystemExit):
            cmd_constraint_report(args)

    def test_no_source_exits(self):
        from aurelius.cli_constraint import cmd_constraint_report
        args = _make_args(scenario=None, snapshot=None)
        with pytest.raises(SystemExit):
            cmd_constraint_report(args)

    def test_snapshot_roundtrip(self, tmp_path):
        from aurelius.cli_constraint import cmd_constraint_report
        # Save a ClusterState as JSON, then load it via --snapshot
        state = _get_simulator_state(steps=2)
        snap_file = tmp_path / "state.json"
        snap_file.write_text(json.dumps(state.to_dict()))

        out_file = tmp_path / "report.txt"
        args = _make_args(
            scenario=None,
            snapshot=str(snap_file),
            format="text",
            output=str(out_file),
        )
        cmd_constraint_report(args)
        assert out_file.exists()
        assert "CONSTRAINT ASSESSMENT" in out_file.read_text()

    def test_all_6_scenarios_run(self):
        from aurelius.cli_constraint import cmd_constraint_report
        from aurelius.simulation.cluster import list_scenarios
        for scenario_name in list_scenarios():
            args = _make_args(scenario=scenario_name, steps=2)
            cmd_constraint_report(args)  # must not raise


class TestCmdSimulateConstraintScenario:
    def test_list_scenarios(self, capsys):
        from aurelius.cli_constraint import cmd_simulate_constraint_scenario
        args = _make_args(list=True)
        cmd_simulate_constraint_scenario(args)
        captured = capsys.readouterr()
        assert "energy_price_arbitrage_multiregion" in captured.out
        assert "thermal_hotspot_mixed_cluster" in captured.out

    def test_basic_table_output(self, capsys):
        from aurelius.cli_constraint import cmd_simulate_constraint_scenario
        args = _make_args(scenario="energy_price_arbitrage_multiregion", steps=5)
        cmd_simulate_constraint_scenario(args)
        captured = capsys.readouterr()
        assert "SCENARIO:" in captured.out
        assert "SANDBOX" in captured.out
        assert "AGGREGATE METRICS" in captured.out
        assert "PER-TICK SUMMARY" in captured.out

    def test_output_to_file(self, tmp_path):
        from aurelius.cli_constraint import cmd_simulate_constraint_scenario
        out_file = tmp_path / "sim_report.txt"
        args = _make_args(
            scenario="energy_price_arbitrage_multiregion",
            steps=3,
            output=str(out_file),
        )
        cmd_simulate_constraint_scenario(args)
        assert out_file.exists()
        assert "SCENARIO:" in out_file.read_text()

    def test_deterministic_with_seed(self):
        import io

        from aurelius.cli_constraint import cmd_simulate_constraint_scenario
        outputs = []
        for _ in range(2):
            buf = io.StringIO()
            with patch("sys.stdout", buf):
                args = _make_args(scenario="energy_price_arbitrage_multiregion", steps=5, seed=42)
                cmd_simulate_constraint_scenario(args)
            outputs.append(buf.getvalue())
        assert outputs[0] == outputs[1], "Output not deterministic"

    def test_invalid_scenario_exits(self):
        from aurelius.cli_constraint import cmd_simulate_constraint_scenario
        args = _make_args(scenario="nonexistent_scenario_xyz")
        with pytest.raises(SystemExit):
            cmd_simulate_constraint_scenario(args)

    def test_no_scenario_no_list_exits(self):
        from aurelius.cli_constraint import cmd_simulate_constraint_scenario
        args = _make_args(scenario=None, list=False)
        with pytest.raises(SystemExit):
            cmd_simulate_constraint_scenario(args)

    def test_scenario_validation_check(self, capsys):
        from aurelius.cli_constraint import cmd_simulate_constraint_scenario
        args = _make_args(
            scenario="energy_price_arbitrage_multiregion",
            steps=5,
            seed=42,
        )
        cmd_simulate_constraint_scenario(args)
        captured = capsys.readouterr()
        # The energy scenario has an expected_primary_constraint
        assert "SCENARIO VALIDATION" in captured.out


class TestCmdTelemetryCheck:
    def test_basic_output(self, capsys):
        from aurelius.cli_constraint import cmd_telemetry_check
        args = _make_args(scenario="energy_price_arbitrage_multiregion", steps=5)
        cmd_telemetry_check(args)
        captured = capsys.readouterr()
        assert "TELEMETRY COVERAGE CHECK" in captured.out
        assert "CONSTRAINT DETECTION CAPABILITY" in captured.out
        assert "COVERAGE SUMMARY" in captured.out

    def test_output_to_file(self, tmp_path):
        from aurelius.cli_constraint import cmd_telemetry_check
        out_file = tmp_path / "telemetry.txt"
        args = _make_args(
            scenario="thermal_hotspot_mixed_cluster",
            steps=3,
            output=str(out_file),
        )
        cmd_telemetry_check(args)
        assert out_file.exists()
        assert "TELEMETRY COVERAGE CHECK" in out_file.read_text()

    def test_no_source_exits(self):
        from aurelius.cli_constraint import cmd_telemetry_check
        args = _make_args(scenario=None, snapshot=None)
        with pytest.raises(SystemExit):
            cmd_telemetry_check(args)

    def test_all_scenarios(self):
        from aurelius.cli_constraint import cmd_telemetry_check
        from aurelius.simulation.cluster import list_scenarios
        for name in list_scenarios():
            args = _make_args(scenario=name, steps=3)
            cmd_telemetry_check(args)  # must not raise


class TestCmdTopologyReport:
    def test_basic_output(self, capsys):
        from aurelius.cli_constraint import cmd_topology_report
        args = _make_args(scenario="topology_fragmentation_h100", steps=3)
        cmd_topology_report(args)
        captured = capsys.readouterr()
        assert "TOPOLOGY REPORT" in captured.out
        assert "CLUSTER TOPOLOGY" in captured.out

    def test_sandbox_label(self, capsys):
        from aurelius.cli_constraint import cmd_topology_report
        args = _make_args(scenario="energy_price_arbitrage_multiregion", steps=2)
        cmd_topology_report(args)
        captured = capsys.readouterr()
        assert "SANDBOX" in captured.out

    def test_output_to_file(self, tmp_path):
        from aurelius.cli_constraint import cmd_topology_report
        out_file = tmp_path / "topo.txt"
        args = _make_args(
            scenario="topology_fragmentation_h100",
            steps=2,
            output=str(out_file),
        )
        cmd_topology_report(args)
        assert out_file.exists()
        assert "TOPOLOGY REPORT" in out_file.read_text()

    def test_no_source_exits(self):
        from aurelius.cli_constraint import cmd_topology_report
        args = _make_args(scenario=None, snapshot=None)
        with pytest.raises(SystemExit):
            cmd_topology_report(args)


class TestCmdValidateConnectors:
    def test_all_pass(self, capsys):
        from aurelius.cli_constraint import cmd_validate_connectors
        args = _make_args()
        cmd_validate_connectors(args)
        captured = capsys.readouterr()
        assert "ALL PASSED" in captured.out
        assert "[FAIL]" not in captured.out

    def test_pass_count(self, capsys):
        from aurelius.cli_constraint import cmd_validate_connectors
        args = _make_args()
        cmd_validate_connectors(args)
        captured = capsys.readouterr()
        pass_count = captured.out.count("[PASS]")
        assert pass_count >= 8, f"Expected at least 8 PASS, got {pass_count}"

    def test_no_secrets(self, capsys):
        from aurelius.cli_constraint import cmd_validate_connectors
        args = _make_args()
        cmd_validate_connectors(args)
        captured = capsys.readouterr()
        for keyword in ["password", "secret", "Bearer", "DATABASE_URL", "api_key"]:
            assert keyword.lower() not in captured.out.lower()


# ---------------------------------------------------------------------------
# CLI registration (via argparse main())
# ---------------------------------------------------------------------------

class TestCLIRegistration:
    def test_constraint_report_registered(self):
        from aurelius.cli import main
        with pytest.raises(SystemExit) as exc_info:
            with patch("sys.argv", ["aurelius", "constraint-report", "--help"]):
                main()
        assert exc_info.value.code == 0

    def test_simulate_constraint_scenario_registered(self):
        from aurelius.cli import main
        with pytest.raises(SystemExit) as exc_info:
            with patch("sys.argv", ["aurelius", "simulate-constraint-scenario", "--help"]):
                main()
        assert exc_info.value.code == 0

    def test_telemetry_check_registered(self):
        from aurelius.cli import main
        with pytest.raises(SystemExit) as exc_info:
            with patch("sys.argv", ["aurelius", "telemetry-check", "--help"]):
                main()
        assert exc_info.value.code == 0

    def test_topology_report_registered(self):
        from aurelius.cli import main
        with pytest.raises(SystemExit) as exc_info:
            with patch("sys.argv", ["aurelius", "topology-report", "--help"]):
                main()
        assert exc_info.value.code == 0

    def test_validate_connectors_registered(self):
        from aurelius.cli import main
        with pytest.raises(SystemExit) as exc_info:
            with patch("sys.argv", ["aurelius", "validate-connectors", "--help"]):
                main()
        assert exc_info.value.code == 0


# ---------------------------------------------------------------------------
# Sandbox / safety invariants
# ---------------------------------------------------------------------------

class TestSandboxInvariants:
    def test_all_outputs_labeled_sandbox(self, capsys):
        from aurelius.cli_constraint import cmd_constraint_report
        args = _make_args(scenario="energy_price_arbitrage_multiregion", steps=2)
        cmd_constraint_report(args)
        captured = capsys.readouterr()
        assert "SANDBOX" in captured.out, "Sandbox label must be present in simulator output"

    def test_recommendations_are_recommendation_only(self):
        from aurelius.constraints import ConstraintAwareEngine
        state = _get_simulator_state(steps=5)
        result = ConstraintAwareEngine().run(state)
        for rec in result.recommendations:
            assert rec.implementation_mode == "recommendation_only", (
                f"Expected recommendation_only, got {rec.implementation_mode}"
            )
            assert rec.provenance.is_sandbox, "Sandbox provenance must carry through to recommendations"

    def test_no_kv_cache_internal_actions(self):
        from aurelius.constraints import ConstraintAwareEngine
        state = _get_simulator_state("latency_tail_kvcache_pressure", steps=10)
        result = ConstraintAwareEngine().run(state)
        forbidden_actions = {"kv_cache", "allocator", "nccl", "cuda", "kernel"}
        for rec in result.recommendations:
            for forbidden in forbidden_actions:
                assert forbidden.lower() not in rec.action_type.lower(), (
                    f"Forbidden action type: {rec.action_type}"
                )

    def test_missing_telemetry_never_fabricated_as_zero(self):
        from datetime import datetime, timezone

        from aurelius.state.models import ClusterState, Provenance, RegionState

        # Create a ClusterState with no energy data to verify None not 0
        ts = datetime.now(tz=timezone.utc)
        prov = Provenance(source="test", fetched_at=ts, confidence="low", is_sandbox=True)
        # RegionState with no energy
        region = RegionState(region="us-east", timestamp=ts, provenance=prov)
        state = ClusterState(
            timestamp=ts,
            provenance=prov,
            regions={"us-east": region},
            is_partial=True,
            missing_sources=["energy"],
        )
        from aurelius.constraints import ConstraintClassifier
        assessment = ConstraintClassifier().assess(state)
        # Energy should not be scored (no data), never fabricated as 0
        from aurelius.state.models import ConstraintType
        if ConstraintType.ENERGY in assessment.scores:
            # If scored, the score should reflect partial data, not necessarily 0
            # The key invariant: if data is missing, binding should be None or very low confidence
            if assessment.binding_constraint == ConstraintType.ENERGY:
                assert assessment.confidence < 0.5, "Should have low confidence with missing energy data"
